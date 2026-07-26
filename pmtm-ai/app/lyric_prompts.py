from app.genre_rules import get_genre_rules

TARGET_BARS = 8


def build_user_prompt(
    *,
    bpm: float | None = None,
    bars: int | None = None,
    rhyme_scheme: str | None = None,
    topic: str | None = None,
) -> str:
    g_name, default_lines, min_s, max_s = get_genre_rules(bpm)
    target_lines = bars if bars is not None else default_lines
    target_syllables = f"{min_s}~{max_s}"
    topic_clause = f" 주제는 '{topic}'이며," if topic else ""
    base_prompt = (
        f"{g_name} 랩 가사를 작성해 주세요.{topic_clause} "
        f"정확히 {target_lines}줄로 구성해야 하며, "
        f"줄당 음절 수는 {target_syllables} 범위 내로 조절해 주세요."
    )

    if rhyme_scheme:
        return f"{base_prompt} 추가로, 끝단어의 라임은 반드시 {rhyme_scheme} 스키마를 준수해 주세요."
    return base_prompt


def build_messages(
    *,
    bpm: float | None = None,
    bars: int | None = None,
    rhyme_scheme: str | None = None,
    topic: str | None = None,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": build_user_prompt(
                bpm=bpm, bars=bars, rhyme_scheme=rhyme_scheme, topic=topic
            ),
        },
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages

