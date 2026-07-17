import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from app.lyric_prompts import TARGET_BARS, build_api_messages
from app.paths import DATA_DIR, MODEL_ID, MODELS_DIR, OUTPUTS_DIR, PROJECT_ROOT
from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score

sys.path.append(str(PROJECT_ROOT))

SFT_PATH = str(MODELS_DIR / "sft_rap_qwen")
DATA_PATH = str(DATA_DIR / "merged_final_dataset_analyzed.csv")
OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen")
SAVE_DIR = str(MODELS_DIR / "grpo_rap_qwen")
TOP_N = 200
SMOKE_OUTPUT_DIR = str(OUTPUTS_DIR / "grpo_qwen_smoke")


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


import random

def build_prompts(df: pd.DataFrame) -> list[list[dict[str, str]]]:
    top = df.sort_values("rhyme_density", ascending=False).head(TOP_N)
    prompts = []
    for _, row in top.iterrows():
        scheme = "AAAABBBB"
        prompts.append(
            build_api_messages(
                bpm=float(row["bpm"]),
                bars=TARGET_BARS,
                rhyme_scheme=scheme,
            )
        )
    return prompts


_END_RE = re.compile(r"\[End\]")
_BARS_RE = re.compile(r"\[Verse\s+(\d+)\s*마디\]")
_LINES_RE = re.compile(r"exactly\s+(\d+)\s+lines", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_TEXT_CHAR_RE = re.compile(r"[가-힣A-Za-z0-9]")
NGRAM_REPEAT_PENALTY_WEIGHT = 0.8
SEVERE_NGRAM_REPEAT_RATIO = 0.25
SHORT_LINE_MAX_CHARS = 3
SHORT_LINE_PENALTY_WEIGHT = 0.8
SEVERE_SHORT_LINE_RATIO = 0.25


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
        ln = re.sub(r"\(\d+음절\)\s*$", "", ln)  # 음절 표시 제거
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


def _repeated_ngram_ratio(lines: list[str], min_n: int = 2, max_n: int = 4) -> float:
    counts: dict[tuple[str, ...], int] = {}
    total = 0
    for line in lines:
        words = _TOKEN_RE.findall(line.lower())
        for n in range(min_n, max_n + 1):
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                counts[ngram] = counts.get(ngram, 0) + 1
                total += 1

    if total == 0:
        return 0.0

    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / total


def _line_char_length(line: str) -> int:
    return len(_TEXT_CHAR_RE.findall(line))


def _short_line_ratio(lines: list[str], target_bars: int) -> float:
    if not lines or target_bars <= 0:
        return 1.0
    short_lines = sum(1 for line in lines if _line_char_length(line) <= SHORT_LINE_MAX_CHARS)
    return short_lines / target_bars


def rhyme_reward(completions, prompts=None, **kwargs):
    """지정된 라임 스키마(AAAABBBB) 준수 여부만 정밀하게 채점하여 보상 제공."""
    if prompts is None:
        prompts = [""] * len(completions)
    rewards = []
    
    for prompt, comp in zip(prompts, completions):
        lines = _extract_verse(comp)
        
        # 8마디 분량을 보장해야 라임 계산이 유의미하므로, 
        # 라인 수가 부족하면 강력한 감점
        if len(lines) < 8:
            rewards.append(float(-1.5 + (len(lines) / 8.0) * 0.5))
            continue
            
        # 8마디 이상일 때, 상위 8줄만 사용하여 라임 평가
        eval_lines = lines[:8]
        
        # 중복 라인 감지 (동일 문장 반복으로 라임 꼼수 쓰는 것 방지)
        dup_ratio = 1.0 - len(set(eval_lines)) / 8.0
        
        # AAAABBBB 채점
        # 1~4행 통일도 (1행 기준 2~4행 비교)
        scores_a = [get_line_rhyme_score(eval_lines[0], eval_lines[i]) for i in range(1, 4)]
        # 5~8행 통일도 (5행 기준 6~8행 비교)
        scores_b = [get_line_rhyme_score(eval_lines[4], eval_lines[i]) for i in range(5, 8)]
        
        rhyme_score = (sum(scores_a)/3.0 + sum(scores_b)/3.0) / 2.0
        
        # A와 B가 너무 같은 모음군이면 감점 (다양성 확보)
        cross_similarity = get_line_rhyme_score(eval_lines[0], eval_lines[4])
        if cross_similarity > 0.6:
            rhyme_score -= (cross_similarity - 0.6) * 0.2
        
        # 중복으로 점수를 얻으려는 리워드 해킹 감점
        effective_rhyme = rhyme_score * (1.0 - dup_ratio)
        
        # 최종 보상 산출 (오직 라이밍 점수에 비례)
        r = effective_rhyme
        
        # 중복이 극단적으로 심하면 강한 감점
        if dup_ratio >= 0.3:
            r = min(r, -1.0)
            
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
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=6,
        max_steps=max_steps,
        num_generations=4,
        max_completion_length=160,
        beta=0.04,
        scale_rewards="none",
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

    df = pd.read_csv(DATA_PATH)
    prompts = build_prompts(df)
    dataset = Dataset.from_list([{"prompt": prompt} for prompt in prompts])
    print(f"prompts: {len(prompts)}")

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


def train_grpo(*, trace_finite: bool = False):
    callbacks = [GrpoFiniteTraceCallback()] if trace_finite else None
    trainer, _prompts = _build_grpo_trainer(output_dir=OUTPUT_DIR, callbacks=callbacks)

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
