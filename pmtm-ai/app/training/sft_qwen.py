import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
# pyrefly: ignore [missing-import]
from app.training.train_utils import (
    is_unsloth_available,
    load_unsloth_model_and_tokenizer,
    setup_training_env,
)

# pyrefly: ignore [missing-import]
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# pyrefly: ignore [missing-import]
from datasets import Dataset

from app.paths import DATA_DIR, MODEL_ID, MODELS_DIR, OUTPUTS_DIR

DATA_PATH = str(DATA_DIR / "prepared_dataset.jsonl")
OUTPUT_DIR = str(OUTPUTS_DIR / "sft_qwen")
SAVE_DIR = str(MODELS_DIR / "sft_rap_qwen")
MAX_LENGTH = 768
EVAL_RATIO = 0.1
SEED = 42


def _tokenize_messages(tokenizer, messages: list[dict], max_length: int = MAX_LENGTH) -> dict:
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("SFT sample must contain prompt messages followed by an assistant response")

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    labels = list(encoded["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def _make_data_collator(tokenizer):
    def collate(features):
        model_inputs = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = [
            feature["labels"] + [-100] * (max_len - len(feature["labels"]))
            for feature in features
        ]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate


def train_sft(
    genre: str | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
    save_dir: str | None = None,
    use_unsloth: bool = False,
):
    if genre in ("boombap", "trap"):
        target_data_path = data_path or str(DATA_DIR / f"prepared_dataset_{genre}.jsonl")
        target_output_dir = output_dir or str(OUTPUTS_DIR / f"sft_qwen_{genre}")
        target_save_dir = save_dir or str(MODELS_DIR / f"sft_rap_qwen_{genre}")
    else:
        target_data_path = data_path or DATA_PATH
        target_output_dir = output_dir or OUTPUT_DIR
        target_save_dir = save_dir or SAVE_DIR

    active_unsloth = use_unsloth and is_unsloth_available()
    print(f"[unsloth] requested={use_unsloth}, active={active_unsloth}")

    if active_unsloth:
        print(f"[unsloth] Loading model with FastLanguageModel ({MODEL_ID})...")
        model, tokenizer = load_unsloth_model_and_tokenizer(
            model_id=MODEL_ID,
            max_seq_length=MAX_LENGTH,
            load_in_4bit=True,
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            padding_side="right",
        )

        use_bf16 = False
        use_fp16 = True
        compute_dtype = torch.float16
    else:
        tokenizer, bnb_config, compute_dtype, use_bf16, use_fp16 = setup_training_env(
            MODEL_ID, padding_side="right"
        )
        print(f"[precision] dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}")

    print(f"[dataset] data_path={target_data_path}")
    print(f"[output] output_dir={target_output_dir}")
    print(f"[save] save_dir={target_save_dir}")

    raw = Dataset.from_json(target_data_path)
    if "messages" not in raw.column_names:
        raise ValueError(f"{target_data_path} must contain a 'messages' column. Regenerate it with prepare_dataset.py.")

    split = raw.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_raw, eval_raw = split["train"], split["test"]

    def tokenize_function(examples):
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for messages in examples["messages"]:
            encoded = _tokenize_messages(tokenizer, messages)
            for key in batch:
                batch[key].append(encoded[key])
        return batch

    train_ds = train_raw.map(tokenize_function, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tokenize_function, batched=True, remove_columns=eval_raw.column_names)

    if not active_unsloth:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = _make_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=target_output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=7e-5,
        bf16=use_bf16,
        fp16=use_fp16,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.05,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    import glob
    ckpts = sorted(glob.glob(f"{target_output_dir}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = ckpts[-1] if ckpts else None
    print(f"Starting SFT ({MODEL_ID}) — train={len(train_ds)} eval={len(eval_ds)}")
    if resume:
        print(f"[resume] from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(target_save_dir)
    tokenizer.save_pretrained(target_save_dir)
    print(f"SFT done -> {target_save_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen SFT Training")
    parser.add_argument("--genre", choices=["boombap", "trap"], default=None, help="Train specifically for boombap or trap")
    parser.add_argument("--data-path", default=None, help="Path to jsonl dataset")
    parser.add_argument("--output-dir", default=None, help="Path for training checkpoints")
    parser.add_argument("--save-dir", default=None, help="Path to save final adapter")
    parser.add_argument("--use-unsloth", action="store_true", help="Enable Unsloth acceleration")
    args = parser.parse_args()

    train_sft(
        genre=args.genre,
        data_path=args.data_path,
        output_dir=args.output_dir,
        save_dir=args.save_dir,
        use_unsloth=args.use_unsloth,
    )


