import argparse
import json
from pathlib import Path

from app.inference.device import model_device, move_model_to_device, select_inference_device
from app.inference.generate import DEFAULT_REPETITION_PENALTY, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from app.lyric_prompts import (
    TARGET_BARS,
    build_messages as build_lyric_messages,
    build_user_prompt as build_lyric_user_prompt,
    clean_unsupported_characters,
    find_unsupported_characters,
)
from app.paths import MODEL_ID
from app.rhyme_scoring.rhyme_engine import calculate_rhyme_density

PROMPT_FORMATS = ("auto", "chat", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="PMTM API lyric generation CLI")
    p.add_argument("--bpm", type=float, required=True, help="Track BPM")
    p.add_argument("--topic", default=None, help="Target topic (e.g. 자신감/성공, 사랑, 이별, 삶/성찰)")
    p.add_argument("--adapter", default=None, help="Optional LoRA adapter directory path")
    p.add_argument(
        "--base-model",
        default=MODEL_ID,
        help="Override base model id/path.",
    )
    p.add_argument("--bars", type=int, default=None, help="Target bar count (default: auto based on BPM)")
    p.add_argument("--max-new-tokens", type=int, default=400, help="Maximum generated tokens")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P, help="Top-p sampling")
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_REPETITION_PENALTY, help="Repetition penalty for generation")
    p.add_argument(
        "--num-candidates",
        type=int,
        default=4,
        help="Number of candidates to generate for Best-of-N reward selection (default: 4)",
    )
    p.add_argument(
        "--min-rhyme-density",
        type=float,
        default=0.35,
        help="Minimum rhyme density required to accept generation (default: 0.35)",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum generation retries if quality threshold is not met (default: 5)",
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
    return build_lyric_user_prompt(bpm=args.bpm, bars=args.bars, topic=getattr(args, "topic", None))


def build_messages(args) -> list[dict[str, str]]:
    return build_lyric_messages(bpm=args.bpm, bars=args.bars, topic=getattr(args, "topic", None))


def should_use_chat_template(base_model: str, prompt_format: str) -> bool:
    if prompt_format == "chat":
        return True
    if prompt_format == "raw":
        return False
    return "instruct" in base_model.lower()


def resolve_base_model(adapter_path: Path | None, override: str | None) -> str:
    if override:
        return override

    if adapter_path:
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = (
        str(adapter_path)
        if adapter_path and (adapter_path / "tokenizer_config.json").exists()
        else base_model
    )
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



def generate_candidate_texts(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
    num_candidates: int = 4,
) -> list[str]:
    # pyrefly: ignore [missing-import]
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            num_return_sequences=num_candidates,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for i in range(len(output)):
        gen = output[i][prompt_len:]
        candidates.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return candidates


def generate_text(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float = 1.1,
) -> str:
    cands = generate_candidate_texts(
        tokenizer,
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_candidates=1,
    )
    return cands[0]


import re

def post_process_lyrics(raw_text: str) -> str:
    """
    모델의 날것 출력(raw_text)에서 마디 번호(1., 2.) 및 끝단 음절 수/단위 태그((X음절), (10em) 등)를 
    완벽히 제거하고 지원하지 않는 문자(한자, 특수기호 등)를 정제하여 순수 랩 가사 본문만 깨끗하게 정돈해 주는 헬퍼 함수.
    """
    lines = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+\.\s*", "", line)
        line = re.sub(r"^\(\d+[^)]*\)\s*", "", line)
        line = re.sub(r"\(\d+[^)]*\)\s*$", "", line)
        line = clean_unsupported_characters(line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def score_lyric_completion(raw_text: str, bpm: float | None = None) -> float:
    """
    생성된 가사(raw_text)의 라임 밀도와 구조적 완성도를 채점하여 Reward 점수를 반환합니다.
    (GRPO 학습 시 rhyme_reward 채점 기준과 동일)
    """
    # 허용되지 않는 이상 문자가 포함되어 있는지 체크하여 페널티 부여
    unsupported = find_unsupported_characters(raw_text)
    unsupported_penalty = len(unsupported) * 5.0

    clean_text = post_process_lyrics(raw_text)
    lines = [line.strip() for line in clean_text.split("\n") if line.strip()]
    if not lines:
        return -1.0 - unsupported_penalty

    # 1) 공백/특수문자 제거 정규화 문장 기반 중복 줄 수 계산 및 페널티 부여
    norm_lines = [re.sub(r"[^\w]", "", line) for line in lines if line.strip()]
    dup_count = (len(norm_lines) - len(set(norm_lines))) if norm_lines else 0

    if dup_count == 0:
        dup_penalty = 0.0
    elif dup_count == 1:
        dup_penalty = 0.2
    elif dup_count == 2:
        dup_penalty = 0.5
    elif dup_count == 3:
        dup_penalty = 0.8
    else:
        dup_penalty = 1.2

    # 2) 전체 생성 라인의 라임 밀도 계산
    rhyme_score = calculate_rhyme_density(lines, bpm=bpm)

    final_reward = rhyme_score - dup_penalty - unsupported_penalty
    return float(final_reward)


def select_best_candidate(candidates: list[str], bpm: float | None = None) -> tuple[str, float, list[tuple[str, float]]]:
    """
    생성된 후보 가사(candidates) 리스트 중 reward가 가장 높은 후보와 점수 목록을 반환합니다.
    """
    scored = []
    for cand in candidates:
        reward = score_lyric_completion(cand, bpm=bpm)
        scored.append((cand, reward))

    best_cand, best_reward = max(scored, key=lambda item: item[1])
    return best_cand, best_reward, scored


def main():
    args = parse_args()
    adapter_path = Path(args.adapter).expanduser().resolve() if args.adapter else None
    if adapter_path and not adapter_path.exists():
        raise FileNotFoundError(f"adapter not found: {adapter_path}")

    base_model = resolve_base_model(adapter_path, args.base_model)
    tokenizer, model = build_model(base_model, adapter_path)
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

    num_candidates = max(1, getattr(args, "num_candidates", 4))
    min_rhyme_density = getattr(args, "min_rhyme_density", 0.35)
    max_retries = max(1, getattr(args, "max_retries", 5))
    candidates = []
    best_cand = ""
    best_reward = -float("inf")
    scored = []

    import sys
    for attempt in range(1, max_retries + 1):
        candidates = generate_candidate_texts(
            tokenizer,
            model,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_candidates=num_candidates,
        )
        cand, reward, scored_items = select_best_candidate(candidates, bpm=args.bpm)
        if reward > best_reward or attempt == 1:
            best_cand, best_reward, scored = cand, reward, scored_items

        clean_lines = [line.strip() for line in post_process_lyrics(cand).split("\n") if line.strip()]
        rhyme_density = calculate_rhyme_density(clean_lines, bpm=args.bpm)
        unsupported = find_unsupported_characters(cand)

        if not unsupported and rhyme_density >= min_rhyme_density:
            break

        print(f"[Attempt {attempt}/{max_retries}] Candidate rejected. Retrying...", file=sys.stderr)

    print(f"=== Generated {len(candidates)} Candidates ===", file=sys.stderr)
    for idx, (cand, reward) in enumerate(scored, 1):
        print(f"--- Candidate {idx} (Reward: {reward:+.4f}) ---", file=sys.stderr)
        print(cand, file=sys.stderr)

    print("-" * 60, file=sys.stderr)
    print(f"★ Selected Best Candidate (Reward: {best_reward:+.4f})", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    print(post_process_lyrics(best_cand))



if __name__ == "__main__":
    main()

