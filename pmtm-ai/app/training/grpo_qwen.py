import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from app.lyric_prompts import TARGET_BARS, build_api_messages
from app.paths import MODEL_ID, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score

sys.path.append(str(PROJECT_ROOT))

SFT_PATH = str(MODELS_DIR / "sft_rap_qwen")
OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen")
SAVE_DIR = str(MODELS_DIR / "grpo_rap_qwen")
SMOKE_OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen_smoke")
# 실질적으로 2종류의 프롬프트 텍스트(붐뱁/트랩)만 생성되므로 대표 BPM 2개를 명시
GRPO_BPMS: list[float] = [90.0, 140.0]  # 붐뱁(10~14음절), 트랩(14~18음절)


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


class GrpoRewardStdEarlyStoppingCallback(TrainerCallback):
    """reward_std가 threshold 이하로 patience step 연속 유지되면
    체크포인트를 저장하고 학습을 조기 종료한다.

    프롬프트 종류가 고정된 환경에서 모델이 수렴하면 그룹 내 reward 분산이
    0에 가까워져 scale_rewards='group' 정규화가 불안정해지므로, 해당 시점에
    학습을 멈추는 것이 안전하다.
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
                    f"[early_stop] reward_std converged — "
                    f"saving checkpoint and stopping training."
                )
                control.should_save = True
                control.should_training_stop = True
        else:
            self._count = 0


def _latest_checkpoint(output_dir: str) -> str | None:
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    return ckpts[-1] if ckpts else None


def _detect_precision():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True

def build_prompts(df=None) -> list[list[dict[str, str]]]:
    """GRPO_BPMS의 대표 BPM으로 프롬프트를 생성한다.
    학습 길이는 max_steps로 제어하므로 프롬프트 수는 최소한으로 유지한다.
    """
    return [
        build_api_messages(bpm=bpm, bars=TARGET_BARS)
        for bpm in GRPO_BPMS
    ]


_END_RE = re.compile(r"\[End\]")
_BARS_RE = re.compile(r"\[Verse\s+(\d+)\s*마디\]")
_LINES_RE = re.compile(r"exactly\s+(\d+)\s+lines", re.IGNORECASE)


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
    for pattern in (_LINES_RE, _BARS_RE):
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
    """연속된 라임 블록(AA, AAA, AAAA)의 길이에 따라 차등 채점."""
    try:
        from app.rhyme_scoring.rhyme_engine import calculate_line_scores
    except ImportError:
        from rhyme_engine import calculate_line_scores

    if prompts is None:
        prompts = [""] * len(completions)
    rewards = []
    
    for prompt, comp in zip(prompts, completions):
        lines = _extract_verse(comp)
        
        # 아예 한 줄도 생성하지 못한 극단적인 경우 0점 처리
        if not lines:
            rewards.append(0.0)
            continue
            
        actual_lines = lines[:8]
        actual_len = len(actual_lines)
        
        # 1) 실제 줄들의 중복 비율 계산
        dup_ratio = (1.0 - len(set(actual_lines)) / actual_len) if actual_len > 0 else 0.0
        
        # 2) 연속 라임 점수 계산 (AA: 0.6, AAA: 0.8, AAAA: 1.0, 그 외 0.0)
        if actual_len > 1:
            line_scores, _ = calculate_line_scores(actual_lines)
            rhyme_score = sum(line_scores) / actual_len
        else:
            rhyme_score = 0.0

            
        # 3) 중복 리워드 해킹 방지
        effective_rhyme = rhyme_score * (1.0 - dup_ratio)
        r = effective_rhyme
        
        # 중복이 극단적으로 심하면 강한 감점
        if dup_ratio >= 0.3:
            r = min(r, -1.0)
            
        # 4) 분량이 8마디 미만인 경우 비례 스케일링 (단일 분량 페널티)
        if actual_len < 8:
            r = r * (actual_len / 8.0)
            
        rewards.append(float(r))
        
    return rewards


def load_model(compute_dtype):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    if os.path.exists(SFT_PATH):
        print(f"SFT adapter 로드: {SFT_PATH}")
        model = PeftModel.from_pretrained(base, SFT_PATH, is_trainable=True)
    else:
        print(f"[WARN] SFT 어댑터 없음 ({SFT_PATH}) — 베이스에 새 LoRA 부착")
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
    max_steps: int = -1,
    save_strategy: str = "steps",
    save_steps: int = 50,
    save_total_limit: int | None = 3,
) -> GRPOConfig:
    compute_dtype, use_bf16, use_fp16 = _detect_precision()
    return GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        num_generations=8,
        max_completion_length=300,
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
    max_steps: int = -1,
    callbacks: list[TrainerCallback] | None = None,
    save_strategy: str = "steps",
    save_steps: int = 50,
    save_total_limit: int | None = 3,
) -> tuple[GRPOTrainer, list[list[dict[str, str]]]]:
    compute_dtype, use_bf16, use_fp16 = _detect_precision()
    print(f"[precision] dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts()
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    print(f"prompts: {len(prompts)} (bpms={GRPO_BPMS})")

    model = load_model(compute_dtype)
    _assert_finite("loaded_sft.weights", _trainable_parameter_summary(model))
    _assert_finite("loaded_sft.forward_logits", _forward_logits_summary(model, tokenizer, prompts[0]))

    cfg = _build_grpo_config(
        output_dir=output_dir,
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


def train_grpo(*, trace_finite: bool = False, max_steps: int = 200):
    callbacks: list[TrainerCallback] = [GrpoRewardStdEarlyStoppingCallback()]
    if trace_finite:
        callbacks.append(GrpoFiniteTraceCallback())
    trainer, _prompts = _build_grpo_trainer(output_dir=OUTPUT_DIR, callbacks=callbacks, max_steps=max_steps)

    resume = _latest_checkpoint(OUTPUT_DIR)
    print(f"Starting GRPO ({MODEL_ID})...")
    if trace_finite:
        print("[trace] finite checks enabled for full GRPO training")
    if resume:
        print(f"[resume] from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(SAVE_DIR)
    print(f"GRPO done -> {SAVE_DIR}")


def run_grpo_smoke_test(max_steps: int = 10) -> None:
    if max_steps < 1 or max_steps > 50:
        raise ValueError("GRPO smoke test supports 1 to 50 steps.")

    output_dir = Path(SMOKE_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer, _prompts = _build_grpo_trainer(
        output_dir=str(output_dir),
        max_steps=max_steps,
        callbacks=[GrpoFiniteTraceCallback()],
        save_strategy="no",
        save_total_limit=None,
    )

    print(f"Starting GRPO smoke test ({max_steps} steps, output_dir={output_dir})...")
    trainer.train(resume_from_checkpoint=None)
    _assert_finite("smoke_end.weights", _trainable_parameter_summary(trainer.model))
    print("GRPO smoke test completed without non-finite tensors.")


if __name__ == "__main__":
    train_grpo()
