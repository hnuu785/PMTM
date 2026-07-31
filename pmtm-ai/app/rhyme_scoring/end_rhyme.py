"""End-Rhyme evaluation module.
Evaluates end-of-line syllable rhyming patterns (AA, AABB, ABAB)
and checks whether line endings form clear vowel/syllable rhymes.
"""

from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score


def check_end_rhyme_pair(line1: str, line2: str, threshold: float = 0.5) -> bool:
    """두 문장 끝단어 간의 라임 점수가 임계값(0.5) 이상인지 검증합니다."""
    score = get_line_rhyme_score(line1, line2)
    return score >= threshold


def calculate_chunk_end_rhyme_score(lines: list[str]) -> float:
    """청크 내 각 라인 끝단어가 인접/건너뛴 라인과 명확한 엔드 라임(End-Rhyme)을 이루는지 채점합니다.
    (최소 0.0 ~ 최대 1.0)
    """
    n = len(lines)
    if n <= 1:
        return 0.0

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
