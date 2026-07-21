"""Genre classification and syllable range rules based on BPM."""

def get_genre_and_syllable_range(bpm: float | None) -> tuple[str, int, int]:
    """BPM을 기준으로 장르 이름과 목표 음절 범위(min_s, max_s)를 반환합니다.
    bpm이 None인 경우 기본값으로 트랩(14, 18)을 반환합니다.
    """
    if bpm is None:
        return "트랩", 14, 18
    
    # 60 ~ 80 BPM 범위는 하프타임으로 간주하여 2배로 계산
    judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
    if judgment_bpm < 110:
        return "붐뱁", 10, 14
    else:
        return "트랩", 14, 18
