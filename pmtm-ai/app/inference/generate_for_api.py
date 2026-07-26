import argparse
import json
from pathlib import Path

from app.inference.device import model_device, move_model_to_device, select_inference_device
from app.inference.generate import DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from app.lyric_prompts import TARGET_BARS, build_messages as build_lyric_messages, build_user_prompt as build_lyric_user_prompt
from app.paths import MODEL_ID

PROMPT_FORMATS = ("auto", "chat", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="PMTM API lyric generation CLI")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM")
    p.add_argument("--topic", default=None, help="Target topic (e.g. 자신감/성공, 사랑, 이별, 삶/성찰)")
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
    p.add_argument("--bars", type=int, default=None, help="Target bar count (default: auto based on BPM)")
    p.add_argument("--max-new-tokens", type=int, default=400, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling")
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="Repetition penalty for generation")
    p.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the constructed prompt before generation",
    )
    p.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="auto",
        help="Prompt format. auto uses chat template for Instruct base models and raw text otherwise.",
    )
    return p.parse_args()


def build_prompt(args) -> str:
    return build_lyric_user_prompt(bpm=args.bpm, bars=args.bars, topic=getattr(args, "topic", None))


def build_messages(args) -> list[dict[str, str]]:
    return build_lyric_messages(bpm=args.bpm, bars=args.bars, topic=getattr(args, "topic", None))


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
        # pyrefly: ignore [missing-import]
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
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from peft import PeftModel
    # pyrefly: ignore [missing-import]
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


def generate_text(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float, top_p: float, repetition_penalty: float = 1.1) -> str:
    # pyrefly: ignore [missing-import]
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


import re

def post_process_lyrics(raw_text: str) -> str:
    """
    모델의 날것 출력(raw_text)에서 마디 번호(1., 2.) 및 끝단 음절 수 태그((X음절))를 
    완벽히 제거하여 순수 랩 가사 본문만 깨끗하게 정돈해 주는 헬퍼 함수.
    """
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^\(\d+음절\)\s*", "", line)
        line = re.sub(r"\(\d+음절\)\s*$", "", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


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
    if args.print_prompt:
        import sys
        print("=== Constructed Prompt ===", file=sys.stderr)
        print(prompt, file=sys.stderr)
        print("-" * 60, file=sys.stderr)

    text = generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    
    import sys
    print("=== Raw Generated Output ===", file=sys.stderr)
    print(text, file=sys.stderr)
    print(post_process_lyrics(text))


if __name__ == "__main__":
    main()
