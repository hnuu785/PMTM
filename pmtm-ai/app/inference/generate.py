import argparse
import json
from pathlib import Path

from app.inference.device import model_device, move_model_to_device, select_inference_device
from app.lyric_prompts import TARGET_BARS, build_messages as build_lyric_messages, build_user_prompt as build_lyric_user_prompt
from app.paths import MODEL_ID, MODELS_DIR

DEFAULT_ADAPTER = MODELS_DIR / "grpo_rap_qwen"
PROMPT_FORMATS = ("auto", "chat", "raw")
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.95
DEFAULT_REPETITION_PENALTY = 1.1


def parse_args():
    p = argparse.ArgumentParser(description="PMTM lyric generation CLI")
    p.add_argument(
        "--adapter",
        default=str(DEFAULT_ADAPTER),
        help="LoRA adapter directory path (default: models/grpo_rap_qwen). Set to 'none' to run the base model only.",
    )
    p.add_argument("--artist", help="Artist style name")
    p.add_argument("--bpm", type=float, help="Track BPM")
    p.add_argument("--energy", type=float, help="Energy score (0-1)")
    p.add_argument("--danceability", type=float, help="Danceability score (0-1)")
    p.add_argument("--loudness", type=float, help="Loudness in dB")
    p.add_argument("--valence", type=float, help="Valence score (0-1)")
    p.add_argument("--bars", type=int, default=None, help="Target bar count (default: auto based on BPM)")
    p.add_argument("--max-new-tokens", type=int, default=400, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling")
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="Repetition penalty for generation")
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
    p.add_argument(
        "--prompt-format",
        choices=PROMPT_FORMATS,
        default="auto",
        help="Prompt format. auto uses chat template for Instruct base models and raw text otherwise.",
    )
    return p.parse_args()


def build_prompt(args) -> str:
    return build_lyric_user_prompt(bpm=args.bpm, bars=args.bars)


def build_messages(args) -> list[dict[str, str]]:
    if args.bpm is None:
        # Default prompt
        return [
            {
                "role": "user",
                "content": build_lyric_user_prompt(bpm=None, bars=args.bars)
            }
        ]

    # Full structured API style messages
    return build_lyric_messages(bpm=args.bpm, bars=args.bars)


def should_use_chat_template(base_model: str, prompt_format: str) -> bool:
    if prompt_format == "chat":
        return True
    if prompt_format == "raw":
        return False
    return "instruct" in base_model.lower()


def resolve_base_model(adapter_path: Path | None, override: str | None) -> str:
    if override:
        return override

    if adapter_path is None:
        return MODEL_ID

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

    return MODEL_ID


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


def build_model(base_model: str, adapter_path: Path | None):
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    from peft import PeftModel
    # pyrefly: ignore [missing-import]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_path = str(adapter_path) if adapter_path else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device, dtype = select_inference_device(torch)
    if device == "cuda":
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
            dtype=dtype,
            trust_remote_code=True,
        )

    if adapter_path:
        model = PeftModel.from_pretrained(base, str(adapter_path))
    else:
        model = base
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
            remove_invalid_values=True,
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
    if args.adapter.lower() == "none":
        adapter_path = None
    else:
        adapter_path = Path(args.adapter).expanduser().resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter not found: {adapter_path}")

    base_model = resolve_base_model(adapter_path, args.base_model)
    raw_prompt = build_prompt(args)
    messages = build_messages(args)

    tokenizer, model = build_model(base_model, adapter_path)
    prompt = build_model_input_text(tokenizer, base_model, args.prompt_format, raw_prompt, messages)

    if args.print_prompt:
        print(prompt)
        print("-" * 60)

    text = generate_text(
        tokenizer,
        model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    
    print("=== Raw Generated Output ===")
    print(text)
    print("\n=== Post-Processed Cleaned Lyrics ===")
    print(post_process_lyrics(text))


if __name__ == "__main__":
    main()
