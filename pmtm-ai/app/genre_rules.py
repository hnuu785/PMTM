"""Genre classification, target line count, and syllable range rules based on BPM."""

def get_genre_rules(bpm: float | None) -> tuple[str, int, int, int]:
    """BPM을 기준으로 장르 이름, 목표 줄 수(target_lines), 목표 음절 범위(min_s, max_s)를 반환합니다.
    - 붐뱁 (< 115 BPM): 8줄, 6~18음절
    - 트랩 (>= 115 BPM): 16줄, 6~18음절
    """
    if bpm is None:
        return "트랩", 16, 6, 18

    judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
    if judgment_bpm < 115:
        return "붐뱁", 8, 6, 18
    else:
        return "트랩", 16, 6, 18


def get_genre_and_syllable_range(bpm: float | None) -> tuple[str, int, int]:
    """기존 호환용: (장르명, min_s, max_s) 반환"""
    genre, _, min_s, max_s = get_genre_rules(bpm)
    return genre, min_s, max_s


