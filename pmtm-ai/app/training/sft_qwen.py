import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from app.training.train_utils import setup_training_env
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from app.paths import DATA_DIR, MODEL_ID, MODELS_DIR, OUTPUTS_DIR

DATA_PATH = str(DATA_DIR / "prepared_dataset_v3.jsonl")
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


def train_sft():
    tokenizer, bnb_config, compute_dtype, use_bf16, use_fp16 = setup_training_env(
        MODEL_ID, padding_side="right"
    )
    print(f"[precision] dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}")

    raw = Dataset.from_json(DATA_PATH)
    if "messages" not in raw.column_names:
        raise ValueError("prepared_dataset_v2.jsonl must contain a 'messages' column. Regenerate it with prepare_dataset_v2.py.")
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # gradient_checkpointing 끔 (fast variant): 1.5B + 4bit + LoRA(r=32)는 T4 16GB에 들어감.
    # backward에서 activation 재계산을 건너뛰어 약 1.5~2배 속도 향상.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = _make_data_collator(tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=125,
        save_total_limit=3,
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
    ckpts = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"),
                   key=lambda p: int(p.rsplit("-", 1)[-1]))
    resume = ckpts[-1] if ckpts else None
    print(f"Starting SFT ({MODEL_ID}) — train={len(train_ds)} eval={len(eval_ds)}")
    if resume:
        print(f"[resume] from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"SFT done -> {SAVE_DIR}")


if __name__ == "__main__":
    train_sft()
