TARGET_BARS = 8
DEFAULT_GENRE = "Korean hip-hop"
DEFAULT_MOOD = "confident"


def build_api_user_prompt(
    *,
    bpm: float | None = None,
    genre: str = DEFAULT_GENRE,
    mood: str = DEFAULT_MOOD,
    bars: int = TARGET_BARS,
) -> str:
    bpm_prompt = f"BPM {bpm:.0f}. " if bpm is not None else ""
    return (
        f"{bpm_prompt}Write exactly {bars} lines of Korean rap verse. "
        "Use natural rhymes and consistent line breathing. "
        "Avoid repeating the same words, phrases, or ending words."
    )


def build_api_messages(
    *,
    bpm: float | None = None,
    genre: str = DEFAULT_GENRE,
    mood: str = DEFAULT_MOOD,
    bars: int = TARGET_BARS,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "user",
            "content": build_api_user_prompt(bpm=bpm, genre=genre, mood=mood, bars=bars),
        },
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages
