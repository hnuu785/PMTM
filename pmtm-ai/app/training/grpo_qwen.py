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


def build_prompts(df: pd.DataFrame) -> list[str]:
    top = df.sort_values("rhyme_density", ascending=False).head(TOP_N)
    prompts = []
    for _, row in top.iterrows():
        artist = row["artist"]
        audio = (
            f"BPM: {row['bpm']:.0f} | "
            f"에너지: {row['energy']:.2f} | "
            f"댄서빌리티: {row['danceability']:.2f} | "
            f"라우드니스: {row['loudness']:.1f}dB | "
            f"밸런스: {row['valence']:.2f}"
        )
        for n_bars in (8, 16):
            prompts.append(f"아티스트: {artist}\n{audio}\n[Verse {n_bars}마디]\n")
    return prompts


_END_RE = re.compile(r"\[End\]")
_BARS_RE = re.compile(r"\[Verse\s+(\d+)\s*마디\]")


def _extract_verse(completion: str) -> list[str]:
    body = _END_RE.split(completion, 1)[0]
    return [ln.strip() for ln in body.split("\n") if ln.strip()]


def _parse_target_bars(prompt: str, default: int = 8) -> int:
    m = _BARS_RE.search(prompt or "")
    return int(m.group(1)) if m else default


def rhyme_reward(completions, prompts=None, **kwargs):
    """라임 0.6 + 길이 0.2 + format 0.2 − 0.5*반복비율."""
    if prompts is None:
        prompts = [""] * len(completions)
    rewards = []
    for prompt, comp in zip(prompts, completions):
        lines = _extract_verse(comp)
        n_target = _parse_target_bars(prompt)

        if len(lines) < 2:
            rhyme = 0.0
        else:
            scores = [get_line_rhyme_score(lines[i], lines[i + 1])
                      for i in range(len(lines) - 1)]
            rhyme = sum(scores) / len(scores)

        format_ok = 1.0 if "[End]" in comp else 0.0
        length_score = max(0.0, 1.0 - abs(len(lines) - n_target) / n_target)
        dup_ratio = 1.0 - len(set(lines)) / len(lines) if lines else 1.0

        r = 0.6 * rhyme + 0.2 * length_score + 0.2 * format_ok - 0.5 * dup_ratio
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
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"prompts: {len(prompts)}")

    model = load_model(compute_dtype)

    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
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
