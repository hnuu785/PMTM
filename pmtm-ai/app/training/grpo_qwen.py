import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import re
import sys

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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


def build_prompts(df: pd.DataFrame) -> list[list[dict[str, str]]]:
    top = df.sort_values("rhyme_density", ascending=False).head(TOP_N)
    prompts = []
    for _, row in top.iterrows():
        prompts.append(
            build_api_messages(
                bpm=float(row["bpm"]),
                bars=TARGET_BARS,
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
    body = _END_RE.split(_message_content(completion), 1)[0]
    return [ln.strip() for ln in body.split("\n") if ln.strip()]


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
    """반복 가사가 라임 점수를 부풀리지 못하도록 보상 계산."""
    if prompts is None:
        prompts = [""] * len(completions)
    rewards = []
    for prompt, comp in zip(prompts, completions):
        completion_text = _message_content(comp)
        lines = _extract_verse(comp)
        n_target = _parse_target_bars(prompt)

        if len(lines) < 2:
            rhyme = 0.0
        else:
            scores = [get_line_rhyme_score(lines[i], lines[i + 1])
                      for i in range(len(lines) - 1)]
            rhyme = sum(scores) / len(scores)

        format_ok = 1.0 if "[End]" in completion_text else 0.0
        length_score = max(0.0, 1.0 - abs(len(lines) - n_target) / n_target)
        dup_ratio = 1.0 - len(set(lines)) / len(lines) if lines else 1.0
        max_run = _max_consecutive_duplicate_run(lines)
        run_penalty = max(0.0, (max_run - 1) / max(1, len(lines) - 1)) if lines else 1.0
        ngram_repeat_ratio = _repeated_ngram_ratio(lines)
        short_line_ratio = _short_line_ratio(lines, n_target)

        # 반복이 많을수록 라임 보상 자체를 줄여서 "반복 = 고득점"을 차단한다.
        repetition_pressure = max(dup_ratio, ngram_repeat_ratio)
        effective_rhyme = rhyme * (1.0 - repetition_pressure)
        r = (
            0.5 * effective_rhyme
            + 0.2 * length_score
            + 0.2 * format_ok
            - 1.0 * dup_ratio
            - 1.0 * run_penalty
            - NGRAM_REPEAT_PENALTY_WEIGHT * ngram_repeat_ratio
            - SHORT_LINE_PENALTY_WEIGHT * short_line_ratio
        )

        # 중복이 심하거나 연속 중복이 발생한 completion은 최종 보상 상한을 강제로 크게 낮춘다.
        if (
            dup_ratio >= 0.3
            or max_run >= 2
            or ngram_repeat_ratio >= SEVERE_NGRAM_REPEAT_RATIO
            or short_line_ratio >= SEVERE_SHORT_LINE_RATIO
        ):
            r = min(r, -1.5)

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


def train_grpo():
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

    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=6,
        num_generations=4,
        max_completion_length=160,
        beta=0.04,
        temperature=1.0,
        top_p=0.95,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_steps=5,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=rhyme_reward,
        args=cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    resume = _latest_checkpoint(OUTPUT_DIR)
    print(f"Starting GRPO ({MODEL_ID})...")
    if resume:
        print(f"[resume] from {resume}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(SAVE_DIR)
    print(f"GRPO done -> {SAVE_DIR}")


if __name__ == "__main__":
    train_grpo()
