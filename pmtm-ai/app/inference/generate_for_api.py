import argparse
import json
from pathlib import Path

from app.paths import MODEL_ID


def parse_args():
    p = argparse.ArgumentParser(description="PMTM API lyric generation CLI")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter directory path")
    p.add_argument(
        "--base-model",
        default=MODEL_ID,
        help="Override base model id/path.",
    )
    p.add_argument(
        "--tokenizer-model",
        default=None,
        help="Override tokenizer id/path. Useful when the base model cache lacks tokenizer files.",
    )
    p.add_argument("--bars", type=int, choices=[8], default=8, help="Target bar count")
    p.add_argument("--max-new-tokens", type=int, default=180, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.92, help="Top-p sampling")
    return p.parse_args()


def build_prompt(args) -> str:
    return (
        "You write rap lyrics. Return only an 8-line verse. "
        "Do not include title, explanation, numbering, or markdown.\n"
        f"Write an 8-bar rap verse for BPM {args.bpm:.0f}. "
        "Each line should feel like one bar and match the BPM's breathing and line length.\n"
        f"BPM: {args.bpm:.0f}\n"
        f"[Verse {args.bars}마디]\n"
    )


def resolve_base_model(adapter_path: Path | None, override: str) -> str:
    if not adapter_path:
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

    return override


def build_model(base_model: str, adapter_path: Path | None, tokenizer_model: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = tokenizer_model or (str(adapter_path) if adapter_path else base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))

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
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    base_model = resolve_base_model(adapter_path, args.base_model)
    tokenizer, model = build_model(base_model, adapter_path, args.tokenizer_model)
    text = generate_text(
        tokenizer,
        model,
        build_prompt(args),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)


if __name__ == "__main__":
    main()
