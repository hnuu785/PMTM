import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from datasets import Dataset
# pyrefly: ignore [missing-import]
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM
# pyrefly: ignore [missing-import]
from transformers import TrainerCallback
from app.training.train_utils import setup_training_env
# pyrefly: ignore [missing-import]
from trl import GRPOConfig, GRPOTrainer

from app.lyric_prompts import TARGET_BARS, build_messages
from app.paths import MODEL_ID, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from app.rhyme_scoring.rhyme_engine import calculate_rhyme_density

sys.path.append(str(PROJECT_ROOT))

SFT_PATH = str(MODELS_DIR / "sft_rap_qwen")
OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen")
SAVE_DIR = str(MODELS_DIR / "grpo_rap_qwen")
SMOKE_OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen_smoke")
# 실질적으로 2종류의 프롬프트 텍스트(붐뱁/트랩)만 생성되므로 대표 BPM 2개를 명시
GRPO_BPMS: list[float] = [90.0, 140.0]  # 붐뱁(8~16음절), 트랩(6~14음절)
GRPO_TOPICS: list[str | None] = [
    None,
    "자신감/성공",
    "삶/성찰",
    "사랑/이별",
]


@dataclass
class FiniteSummary:
    total: int
    nonfinite: int
    max_abs: float
    examples: list[str]

    @property
    def ok(self) -> bool:
        return self.nonfinite == 0


def _summarize_tensors(named_tensors, *, max_examples: int = 5) -> FiniteSummary:
    total = 0
    nonfinite = 0
    max_abs = 0.0
    examples: list[str] = []

    for name, tensor in named_tensors:
        if tensor is None:
            continue
        data = tensor.detach().float()
        total += data.numel()
        finite_mask = torch.isfinite(data)
        bad_count = data.numel() - int(finite_mask.sum().item())
        nonfinite += bad_count
        if bad_count and len(examples) < max_examples:
            examples.append(name)
        if finite_mask.any():
            max_abs = max(max_abs, float(data[finite_mask].abs().max().item()))

    return FiniteSummary(
        total=total,
        nonfinite=nonfinite,
        max_abs=max_abs,
        examples=examples,
    )


def _format_finite_summary(label: str, summary: FiniteSummary) -> str:
    status = "OK" if summary.ok else "BAD"
    return (
        f"[finite:{label}] {status} "
        f"total={summary.total} nonfinite={summary.nonfinite} "
        f"max_abs={summary.max_abs:.6g} examples={summary.examples}"
    )


def _trainable_parameter_summary(model) -> FiniteSummary:
    return _summarize_tensors(
        (name, param.data)
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def _gradient_summary(model) -> FiniteSummary:
    return _summarize_tensors(
        (name, param.grad)
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    )


def _assert_finite(label: str, summary: FiniteSummary) -> None:
    print(_format_finite_summary(label, summary))
    if not summary.ok:
        raise RuntimeError(f"Non-finite tensor detected during {label}: {summary.examples}")


def _model_device(model):
    return next(model.parameters()).device


def _forward_logits_summary(model, tokenizer, prompt: list[dict[str, str]]) -> FiniteSummary:
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(_model_device(model))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    if was_training:
        model.train()
    return _summarize_tensors([("logits", outputs.logits)])


class GrpoFiniteTraceCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            _assert_finite("train_begin.weights", _trainable_parameter_summary(model))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            _assert_finite(f"step_{state.global_step}.before_optimizer.gradients", _gradient_summary(model))

    def on_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            _assert_finite(f"step_{state.global_step}.after_optimizer.weights", _trainable_parameter_summary(model))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        bad = {
            key: value
            for key, value in logs.items()
            if isinstance(value, float) and not torch.isfinite(torch.tensor(value))
        }
        if bad:
            raise RuntimeError(f"Non-finite trainer log at step {state.global_step}: {bad}")


class GrpoBestRewardCallback(TrainerCallback):
    """학습 진행 중 reward_mean이 가장 높았던 최고 성적의 모델을 추적하여
    'best_checkpoint' 경로에 저장합니다.
    """

    def __init__(self):
        self.best_reward: float = float("-inf")
        self.best_checkpoint: str | None = None
        self.best_step: int = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        r = logs.get("reward")
        if r is None:
            r = logs.get("reward_mean")

        if r is not None and r > self.best_reward:
            self.best_reward = float(r)
            self.best_step = state.global_step
            model = kwargs.get("model")
            if model is not None:
                best_dir = os.path.join(args.output_dir, "best_checkpoint")
                model.save_pretrained(best_dir)
                self.best_checkpoint = best_dir
                print(
                    f"[best_model] ★ New Best reward={self.best_reward:+.4f} at step {state.global_step} (saved to {best_dir})"
                )


class GrpoRewardStdEarlyStoppingCallback(TrainerCallback):
    """reward_std가 threshold 이하로 patience step 연속 유지되면
    학습을 조기 종료한다.
    """

    def __init__(self, threshold: float = 0.05, patience: int = 3):
        self.threshold = threshold
        self.patience = patience
        self._count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        reward_std = logs.get("reward_std")
        if reward_std is None:
            return
        if reward_std < self.threshold:
            self._count += 1
            print(
                f"[early_stop] step={state.global_step} "
                f"reward_std={reward_std:.4f} < {self.threshold} "
                f"({self._count}/{self.patience})"
            )
            if self._count >= self.patience:
                print(
                    f"[early_stop] reward_std converged — stopping training."
                )
                control.should_training_stop = True
        else:
            self._count = 0



def _latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def build_prompts(df=None, genre: str | None = None) -> list[list[dict[str, str]]]:
    """GRPO_BPMS 및 대표 TOPIC 조합으로 프롬프트를 생성한다."""
    if genre == "boombap":
        bpms = [90.0]
    elif genre == "trap":
        bpms = [140.0]
    else:
        bpms = GRPO_BPMS

    prompts = []
    for bpm in bpms:
        for topic in GRPO_TOPICS:
            prompts.append(build_messages(bpm=bpm, bars=None, topic=topic))
    return prompts


_END_RE = re.compile(r"\[End\]")
_BARS_RE = re.compile(r"\[Verse\s+(\d+)\s*마디\]")
_LINES_RE = re.compile(r"exactly\s+(\d+)\s+lines", re.IGNORECASE)
_KO_LINES_RE = re.compile(r"정확히\s+(\d+)줄")


def _message_content(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "content" in value:
            return _message_content(value["content"])
        if "text" in value:
            return str(value["text"])
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            content = _message_content(item)
            if content:
                parts.append(content)
        return "\n".join(parts)
    return str(value)


def _extract_verse(completion) -> list[str]:
    text = _message_content(completion)
    if "[수정된 지침에 따른 가사]" in text:
        parts = text.split("[수정된 지침에 따른 가사]")
        body = parts[1] if len(parts) > 1 else text
    else:
        body = text
    body = _END_RE.split(body, 1)[0]
    
    lines = []
    for ln in body.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if not re.match(r"^\d+\.", ln):
            continue
        ln = re.sub(r"^\d+\.\s*", "", ln)  # 마디 번호 제거
        ln = re.sub(r"^\(\d+음절\)\s*", "", ln)  # 음절 표시 제거 (앞단어)
        ln = re.sub(r"\(\d+음절\)\s*$", "", ln)  # 음절 표시 제거 (뒷단어)
        ln = ln.strip()
        if ln:
            lines.append(ln)
    return lines


def _parse_target_bars(prompt, default: int = TARGET_BARS) -> int:
    prompt_text = _message_content(prompt)
    for pattern in (_KO_LINES_RE, _LINES_RE, _BARS_RE):
        m = pattern.search(prompt_text)
        if m:
            return int(m.group(1))
    return default


def _max_consecutive_duplicate_run(lines: list[str]) -> int:
    if not lines:
        return 0

    max_run = 1
    current_run = 1
    for i in range(1, len(lines)):
        if lines[i] == lines[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def rhyme_reward(completions, prompts=None, **kwargs):
    """생성된 가사의 라임 밀도를 산출하고, 중복 줄 수에 따른 단계별 차감 감점을 부여합니다."""
    rewards = []
    
    for idx, comp in enumerate(completions):
        lines = _extract_verse(comp)
        
        # 아예 한 줄도 생성하지 못한 경우 최소 보상 처리
        if not lines:
            rewards.append(-1.0)
            continue
            
        bpm = None
        genre = None
        if prompts and idx < len(prompts):
            p_text = str(prompts[idx])
            m = re.search(r"BPM:\s*(\d+(?:\.\d+)?)", p_text) or re.search(r"bpm[=:]\s*(\d+(?:\.\d+)?)", p_text)
            if m:
                bpm = float(m.group(1))
            elif "트랩" in p_text:
                genre = "트랩"
            elif "붐뱁" in p_text:
                genre = "붐뱁"

        # 1) 공백/특수문자를 제거한 정규화 문장 기반으로 중복 줄 수 계산
        norm_lines = [re.sub(r"[^\w]", "", line) for line in lines if line.strip()]
        dup_count = (len(norm_lines) - len(set(norm_lines))) if norm_lines else 0
        
        # 2) 전체 생성 라인의 평균 라임 점수 계산 (BPM/장르 정보 반영)
        rhyme_score = calculate_rhyme_density(lines, bpm=bpm, genre=genre)

        # 3) 중복 줄 수에 따른 단계별 차감 페널티 (Subtractive Penalty Gradient)
        if dup_count == 0:
            dup_penalty = 0.0
        elif dup_count == 1:
            dup_penalty = 0.2
        elif dup_count == 2:
            dup_penalty = 0.5
        elif dup_count == 3:
            dup_penalty = 0.8
        else:
            dup_penalty = 1.2  # 4줄 이상 중복 도배 시 강력 차단

        final_reward = rhyme_score - dup_penalty
        rewards.append(float(final_reward))
        
    return rewards


from app.training.train_utils import (
    is_unsloth_available,
    load_unsloth_model_and_tokenizer,
    setup_training_env,
)


def load_model(bnb_config, sft_path: str | None = None, genre: str | None = None, use_unsloth: bool = False):
    target_sft_path = sft_path or (str(MODELS_DIR / f"sft_rap_qwen_{genre}") if genre else SFT_PATH)
    active_unsloth = use_unsloth and is_unsloth_available()

    if active_unsloth:
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)
        if os.path.exists(target_sft_path):
            print(f"[unsloth] SFT adapter 로드: {target_sft_path}")
            model, _ = FastLanguageModel.from_pretrained(
                model_name=target_sft_path,
                max_seq_length=2048,
                load_in_4bit=True,
                trust_remote_code=True,
            )
        else:
            print(f"[WARN][unsloth] SFT 어댑터 없음 ({target_sft_path}) — 베이스에 새 LoRA 부착")
            model, _ = load_unsloth_model_and_tokenizer(
                model_id=MODEL_ID,
                max_seq_length=2048,
                load_in_4bit=True,
                r=32,
                lora_alpha=64,
                lora_dropout=0.0,
            )

        return model

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    if os.path.exists(target_sft_path):
        print(f"SFT adapter 로드: {target_sft_path}")
        model = PeftModel.from_pretrained(base, target_sft_path, is_trainable=True)
    else:
        print(f"[WARN] SFT 어댑터 없음 ({target_sft_path}) — 베이스에 새 LoRA 부착")
        lora_config = LoraConfig(
            r=32, lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_config)
    return model


def _build_grpo_config(
    *,
    output_dir: str,
    use_bf16: bool,
    use_fp16: bool,
    max_steps: int = -1,
    save_strategy: str = "steps",
    save_steps: int = 50,
    save_total_limit: int | None = 3,
) -> GRPOConfig:
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        max_steps=max_steps,
        num_generations=8,
        max_completion_length=512,
        beta=0.04,
        scale_rewards="group",  # 그룹 내 std 정규화 → 보상 분포 편향 시 학습 안정화
        cast_lm_head_to_fp32=False,
        temperature=1.0,
        top_p=0.95,
        generation_kwargs={
            "remove_invalid_values": True,
            "renormalize_logits": True,
        },
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        logging_steps=5,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=42,
    )


def _build_grpo_trainer(
    *,
    output_dir: str,
    genre: str | None = None,
    sft_path: str | None = None,
    max_steps: int = -1,
    callbacks: list[TrainerCallback] | None = None,
    save_strategy: str = "steps",
    save_steps: int = 50,
    save_total_limit: int | None = 3,
    use_unsloth: bool = False,
) -> tuple[GRPOTrainer, list[list[dict[str, str]]]]:
    active_unsloth = use_unsloth and is_unsloth_available()
    print(f"[unsloth] requested={use_unsloth}, active={active_unsloth}")

    if active_unsloth:
        from unsloth import FastLanguageModel
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=2048,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        use_bf16 = False
        use_fp16 = True
        bnb_config = None
    else:
        tokenizer, bnb_config, compute_dtype, use_bf16, use_fp16 = setup_training_env(MODEL_ID)
        print(f"[precision] dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}")

    prompts = build_prompts(genre=genre)
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    print(f"prompts: {len(prompts)} (genre={genre})")

    model = load_model(bnb_config, sft_path=sft_path, genre=genre, use_unsloth=use_unsloth)
    _assert_finite("loaded_sft.weights", _trainable_parameter_summary(model))
    _assert_finite("loaded_sft.forward_logits", _forward_logits_summary(model, tokenizer, prompts[0]))

    cfg = _build_grpo_config(
        output_dir=output_dir,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        max_steps=max_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rhyme_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    return trainer, prompts


def train_grpo(
    *,
    genre: str | None = None,
    sft_path: str | None = None,
    output_dir: str | None = None,
    save_dir: str | None = None,
    trace_finite: bool = False,
    max_steps: int = 120,
    use_unsloth: bool = False,
):
    target_output_dir = output_dir or (str(OUTPUTS_DIR / f"grpo_qwen_{genre}") if genre else OUTPUT_DIR)
    target_save_dir = save_dir or (str(MODELS_DIR / f"grpo_rap_qwen_{genre}") if genre else SAVE_DIR)

    best_cb = GrpoBestRewardCallback()
    callbacks: list[TrainerCallback] = [GrpoRewardStdEarlyStoppingCallback(), best_cb]
    if trace_finite:
        callbacks.append(GrpoFiniteTraceCallback())
    trainer, _prompts = _build_grpo_trainer(
        output_dir=target_output_dir,
        genre=genre,
        sft_path=sft_path,
        callbacks=callbacks,
        max_steps=max_steps,
        use_unsloth=use_unsloth,
    )

    resume = _latest_checkpoint(target_output_dir)
    print(f"Starting GRPO ({MODEL_ID}) genre={genre}...")
    if trace_finite:
        print("[trace] finite checks enabled for full GRPO training")
    if resume:
        print(f"[resume] from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # 전체 학습 스텝 중 reward_mean이 가장 높았던 최고 성적 모델을 target_save_dir에 최종 저장
    if best_cb.best_checkpoint and os.path.exists(best_cb.best_checkpoint):
        print(
            f"[final_save] Loading & saving Best Model (reward={best_cb.best_reward:+.4f}, step={best_cb.best_step}) from {best_cb.best_checkpoint} -> {target_save_dir}"
        )
        model = PeftModel.from_pretrained(trainer.model.get_base_model(), best_cb.best_checkpoint)
        model.save_pretrained(target_save_dir)
    else:
        print(f"[final_save] Saving current model state to {target_save_dir}")
        trainer.save_model(target_save_dir)

    print(f"GRPO done -> {target_save_dir}")


def run_grpo_smoke_test(max_steps: int = 10, genre: str | None = None, use_unsloth: bool = False) -> None:
    if max_steps < 1 or max_steps > 50:
        raise ValueError("GRPO smoke test supports 1 to 50 steps.")

    output_path = Path(SMOKE_OUTPUT_DIR if not genre else f"{SMOKE_OUTPUT_DIR}_{genre}")
    output_path.mkdir(parents=True, exist_ok=True)
    trainer, _prompts = _build_grpo_trainer(
        output_dir=str(output_path),
        genre=genre,
        max_steps=max_steps,
        callbacks=[GrpoFiniteTraceCallback()],
        save_strategy="no",
        save_total_limit=None,
        use_unsloth=use_unsloth,
    )

    print(f"Starting GRPO smoke test ({max_steps} steps, output_dir={output_path})...")
    trainer.train(resume_from_checkpoint=None)
    _assert_finite("smoke_end.weights", _trainable_parameter_summary(trainer.model))
    print("GRPO smoke test completed without non-finite tensors.")


if __name__ == "__main__":
    train_grpo()
