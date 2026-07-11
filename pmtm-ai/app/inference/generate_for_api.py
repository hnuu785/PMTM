import argparse
import json
from pathlib import Path

from app.inference.device import model_device, move_model_to_device, select_inference_device
from app.lyric_prompts import TARGET_BARS, build_api_messages, build_api_user_prompt
from app.paths import MODEL_ID

PROMPT_FORMATS = ("auto", "chat", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="PMTM API lyric generation CLI")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM")
    p.add_argument("--genre", default="Korean hip-hop", help="Target genre")
    p.add_argument("--mood", default="confident", help="Target mood")
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
    p.add_argument("--bars", type=int, choices=[TARGET_BARS], default=TARGET_BARS, help="Target bar count")
    p.add_argument("--max-new-tokens", type=int, default=180, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=0.85, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.92, help="Top-p sampling")
    p.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="auto",
        help="Prompt format. auto uses chat template for Instruct base models and raw text otherwise.",
    )
    return p.parse_args()


def build_prompt(args) -> str:
    return build_api_user_prompt(bpm=args.bpm, bars=args.bars)


def build_messages(args) -> list[dict[str, str]]:
    return build_api_messages(bpm=args.bpm, bars=args.bars)


def should_use_chat_template(base_model: str, prompt_format: str) -> bool:
    if prompt_format == "chat":
        return True
    if prompt_format == "raw":
        return False
    return "instruct" in base_model.lower()


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


def build_model_input_text(tokenizer, base_model: str, prompt_format: str, prompt: str, messages: list[dict[str, str]]) -> str:
    if not should_use_chat_template(base_model, prompt_format):
        return prompt

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(f"tokenizer has no chat_template: {getattr(tokenizer, 'name_or_path', base_model)}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


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

    device, dtype = select_inference_device(torch)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model = move_model_to_device(model, device)

    model.eval()
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
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
    prompt = build_model_input_text(
        tokenizer,
        base_model,
        args.prompt_format,
        build_prompt(args),
        build_messages(args),
    )
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
