TARGET_BARS = 8
DEFAULT_GENRE = "Korean hip-hop"
DEFAULT_MOOD = "confident"


def build_api_user_prompt(
    *,
    bpm: float | None = None,
    genre: str = DEFAULT_GENRE,
    mood: str = DEFAULT_MOOD,
    bars: int = TARGET_BARS,
    rhyme_scheme: str | None = None,
) -> str:
    g_normalized = genre.strip().lower()
    if bpm is not None:
        judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
        g_name = "붐뱁" if judgment_bpm < 110 else "트랩"
    elif "boom" in g_normalized or "붐뱁" in g_normalized:
        g_name = "붐뱁"
    else:
        g_name = "트랩"

    target_syllables = "10~14" if g_name == "붐뱁" else "14~18"
    base_prompt = (
        f"랩을 작성해 주세요. "
        f"한 줄당 한 마디(1 Bar) 규칙을 지켜 정확히 {bars}마디로 구성해야 하며, "
        f"마디당 음절 수는 {target_syllables} 범위 내로 조절해 주세요."
    )

    if rhyme_scheme:
        return f"{base_prompt} 추가로, 끝단어의 라임은 반드시 {rhyme_scheme} 스키마를 준수해 주세요."
    return base_prompt


def build_api_messages(
    *,
    bpm: float | None = None,
    genre: str = DEFAULT_GENRE,
    mood: str = DEFAULT_MOOD,
    bars: int = TARGET_BARS,
    rhyme_scheme: str | None = None,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": build_api_user_prompt(
                bpm=bpm, genre=genre, mood=mood, bars=bars, rhyme_scheme=rhyme_scheme
            ),
        },
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages
