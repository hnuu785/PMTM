import argparse
import json
from pathlib import Path

from app.paths import MODEL_ID, MODELS_DIR

DEFAULT_ADAPTER = MODELS_DIR / "grpo_rap_qwen"


def parse_args():
    p = argparse.ArgumentParser(description="PMTM lyric generation CLI")
    p.add_argument(
        "--adapter",
        default=str(DEFAULT_ADAPTER),
        help="LoRA adapter directory path (default: models/grpo_rap_qwen)",
    )
    p.add_argument("--artist", required=True, help="Artist style name")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM")
    p.add_argument("--energy", type=float, required=True, help="Energy score (0-1)")
    p.add_argument("--danceability", type=float, required=True, help="Danceability score (0-1)")
    p.add_argument("--loudness", type=float, required=True, help="Loudness in dB")
    p.add_argument("--valence", type=float, required=True, help="Valence score (0-1)")
    p.add_argument("--bars", type=int, choices=[8, 16], default=8, help="Target bar count")
    p.add_argument("--max-new-tokens", type=int, default=220, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    p.add_argument(
        "--base-model",
        default=None,
        help="Override base model id/path. Defaults to adapter config or PMTM_MODEL_ID.",
    )
    p.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the constructed prompt before generation",
    )
    return p.parse_args()


def build_prompt(args) -> str:
    return (
        f"아티스트: {args.artist}\n"
        f"BPM: {args.bpm:.0f} | "
        f"에너지: {args.energy:.2f} | "
        f"댄서빌리티: {args.danceability:.2f} | "
        f"라우드니스: {args.loudness:.1f}dB | "
        f"밸런스: {args.valence:.2f}\n"
        f"[Verse {args.bars}마디]\n"
    )


def resolve_base_model(adapter_path: Path, override: str | None) -> str:
    if override:
        return override

    try:
        from peft import PeftConfig

        cfg = PeftConfig.from_pretrained(str(adapter_path))
        if cfg.base_model_name_or_path:
            return cfg.base_model_name_or_path
    except Exception:
        pass

    config_path = adapter_path / "adapter_config.json"
    if config_path.exists():
        with config_path.open(encoding="utf-8") as fp:
            config = json.load(fp)
        base_model = config.get("base_model_name_or_path")
        if base_model:
            return str(base_model)

    return MODEL_ID


def build_model(base_model: str, adapter_path: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    adapter_path = Path(args.adapter).expanduser().resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    base_model = resolve_base_model(adapter_path, args.base_model)
    prompt = build_prompt(args)

    if args.print_prompt:
        print(prompt)
        print("-" * 60)

    tokenizer, model = build_model(base_model, adapter_path)
    text = generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
