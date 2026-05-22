import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from app.paths import DATA_DIR, MODEL_ID, MODELS_DIR, OUTPUTS_DIR

DATA_PATH = str(DATA_DIR / "prepared_dataset.jsonl")
OUTPUT_DIR = str(OUTPUTS_DIR / "sft_qwen")
SAVE_DIR = str(MODELS_DIR / "sft_rap_qwen")
MAX_LENGTH = 768
EVAL_RATIO = 0.1
SEED = 42


def _detect_precision():
    """A100/L4/H100 → bf16; T4/V100 → fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False
    return torch.float16, False, True


def train_sft():
    compute_dtype, use_bf16, use_fp16 = _detect_precision()
    print(f"[precision] dtype={compute_dtype}, bf16={use_bf16}, fp16={use_fp16}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = Dataset.from_json(DATA_PATH)
    split = raw.train_test_split(test_size=EVAL_RATIO, seed=SEED)
    train_raw, eval_raw = split["train"], split["test"]

    def tokenize_function(examples):
        texts = [t + tokenizer.eos_token for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=MAX_LENGTH)

    train_ds = train_raw.map(tokenize_function, batched=True, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(tokenize_function, batched=True, remove_columns=eval_raw.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

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
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
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
