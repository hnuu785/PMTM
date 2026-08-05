"""End-Rhyme evaluation module.
Evaluates end-of-line syllable rhyming patterns (AA, AABB, ABAB)
and checks whether line endings form clear vowel/syllable rhymes.
"""

from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score


def check_end_rhyme_pair(line1: str, line2: str, threshold: float = 0.5) -> bool:
    """두 문장 끝단어 간의 라임 점수가 임계값(0.5) 이상인지 검증합니다."""
    score = get_line_rhyme_score(line1, line2)
    return score >= threshold


def calculate_chunk_end_rhyme_score(
    lines: list[str],
    bpm: float | None = None,
    genre: str | None = None,
) -> float:
    """청크 내 각 라인 끝단어가 인접/건너뛴 라인과 명확한 엔드 라임(End-Rhyme)을 이루는지 채점합니다.
    (최소 0.0 ~ 최대 1.0)
    트랩 장르이고 10줄 이상(16줄 구조, 2줄 = 1마디)인 경우, 마디의 끝이 되는 짝수번째 라인을 기준으로 채점합니다.
    """
    n = len(lines)
    if n <= 1:
        return 0.0

    is_trap = False
    if genre is not None:
        is_trap = genre.lower() in ("trap", "트랩")
    elif bpm is not None:
        judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
        is_trap = judgment_bpm >= 115

    if is_trap and n >= 10:
        even_lines = [lines[i] for i in range(1, n, 2)]
        return calculate_chunk_end_rhyme_score(even_lines, bpm=bpm, genre="붐뱁")

    rhyme_matched_lines = 0

    for i in range(n):
        has_rhyme = False
        # 1) 이전/다음 인접 행 검사 (i-1, i+1)
        if i > 0 and check_end_rhyme_pair(lines[i], lines[i - 1]):
            has_rhyme = True
        elif i < n - 1 and check_end_rhyme_pair(lines[i], lines[i + 1]):
            has_rhyme = True
        # 2) 한 줄 건너뛴 행 검사 (i-2, i+2)
        elif i > 1 and check_end_rhyme_pair(lines[i], lines[i - 2]):
            has_rhyme = True
        elif i < n - 2 and check_end_rhyme_pair(lines[i], lines[i + 2]):
            has_rhyme = True

        if has_rhyme:
            rhyme_matched_lines += 1

    return round(rhyme_matched_lines / n, 4)

