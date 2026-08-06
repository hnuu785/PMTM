import re
from app.genre_rules import get_genre_rules

TARGET_BARS = 8

# Allowed lyric characters: Korean, English, Digits, Whitespace, and basic punctuation
ALLOWED_LYRIC_CHARS_REGEX = r"[가-힣a-zA-Z0-9\s.,!?~'\"()\[\]{}:;·…-]"


def find_unsupported_characters(text: str) -> str:
    """가사에서 허용되지 않는 문자(한글, 영문, 숫자, 기본 문장부호 이외)를 추출하여 반환"""
    return re.sub(ALLOWED_LYRIC_CHARS_REGEX, "", text)


def clean_unsupported_characters(text: str) -> str:
    """가사에서 허용되지 않는 문자를 제거한 깨끗한 문자열 반환"""
    return "".join(re.findall(ALLOWED_LYRIC_CHARS_REGEX, text))


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
        f"한국어 중심의 {g_name} 랩 가사를 작성해 주세요.{topic_clause} "
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
