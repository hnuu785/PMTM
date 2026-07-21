from app.genre_rules import get_genre_and_syllable_range

TARGET_BARS = 8


def build_user_prompt(
    *,
    bpm: float | None = None,
    bars: int = TARGET_BARS,
    rhyme_scheme: str | None = None,
) -> str:
    g_name, min_s, max_s = get_genre_and_syllable_range(bpm)
    target_syllables = f"{min_s}~{max_s}"
    base_prompt = (
        f"랩을 작성해주세요. 한 줄당 한 마디 규칙을 지켜 {bars}마디로 구성해야 하며, "
        f"마디 당 음절 수는 {target_syllables} 범위 내로 조절해주세요. "
        f"각 마디 끝 단어 끼리는 라임이 있게 해주세요."
    )

    if rhyme_scheme:
        return f"{base_prompt} 추가로, 끝단어의 라임은 반드시 {rhyme_scheme} 스키마를 준수해 주세요."
    return base_prompt


def build_messages(
    *,
    bpm: float | None = None,
    bars: int = TARGET_BARS,
    rhyme_scheme: str | None = None,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": build_user_prompt(
                bpm=bpm, bars=bars, rhyme_scheme=rhyme_scheme
            ),
        },
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages
