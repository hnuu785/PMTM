"""Smart line structuring module.
Splits over-length lines at word (space) boundaries such that both split lines
fall within acceptable syllable ranges while maximizing end-rhyme scores.
"""

from app.rhyme_scoring.phonetics_utils import count_syllables
from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score


def split_line_by_rhyme(
    line: str,
    min_allowed: int = 5,
    max_allowed: int = 15,
    prev_line: str = "",
    next_line: str = "",
) -> list[str]:
    """긴 가사 라인을 띄어쓰기(어절) 단위로 분할하되,
    1. 분할된 각 줄의 음절 수가 [min_allowed, max_allowed] 안정 범위에 들어오도록 함
    2. 분할된 줄 간 및 인접 줄과의 끝음절 라임 점수(End-Rhyme Score)가 가장 높은 위치를 선택
    3. 필요시 재귀적 분할 수행
    """
    line = line.strip()
    if not line:
        return []

    words = line.split()
    if len(words) <= 1:
        return [line]

    syl_count = count_syllables(line)
    if syl_count <= max_allowed:
        return [line]

    candidates = []

    for k in range(1, len(words)):
        l1 = " ".join(words[:k])
        l2 = " ".join(words[k:])

        syl1 = count_syllables(l1)
        syl2 = count_syllables(l2)

        # 1차 스크리닝: 분할된 라인들이 수용 가능한 음절 수 범위에 있는지 확인
        if (min_allowed - 2 <= syl1 <= max_allowed + 2) and (min_allowed - 2 <= syl2 <= max_allowed + 2):
            score_l1_l2 = get_line_rhyme_score(l1, l2)
            score_prev = get_line_rhyme_score(prev_line, l1) if prev_line else 0.0
            score_next = get_line_rhyme_score(l2, next_line) if next_line else 0.0

            # 라이밍 점수 가중 결합
            total_rhyme_score = score_l1_l2 * 2.0 + score_prev + score_next
            balance_penalty = abs(syl1 - syl2) * 0.02

            candidates.append({
                "k": k,
                "l1": l1,
                "l2": l2,
                "syl1": syl1,
                "syl2": syl2,
                "score": total_rhyme_score - balance_penalty,
            })

    if candidates:
        best = max(candidates, key=lambda c: c["score"])
        res1 = split_line_by_rhyme(best["l1"], min_allowed, max_allowed, prev_line, best["l2"])
        res2 = split_line_by_rhyme(best["l2"], min_allowed, max_allowed, res1[-1] if res1 else "", next_line)
        return res1 + res2
    else:
        mid = len(words) // 2
        l1 = " ".join(words[:mid])
        l2 = " ".join(words[mid:])
        res1 = split_line_by_rhyme(l1, min_allowed, max_allowed, prev_line, l2)
        res2 = split_line_by_rhyme(l2, min_allowed, max_allowed, res1[-1] if res1 else "", next_line)
        return res1 + res2


def merge_underlength_lines(
    lines: list[str],
    min_allowed: int = 5,
    max_allowed: int = 15,
) -> list[str]:
    """음절 수가 너무 짧은 추임새/단문(min_allowed 미만)을 이전 또는 다음 행과 조건부로 병합합니다."""
    if not lines:
        return []

    merged = []
    i = 0
    while i < len(lines):
        line = lines[i]
        syl = count_syllables(line)

        if syl < min_allowed and i + 1 < len(lines):
            next_line = lines[i + 1]
            combined = f"{line} {next_line}"
            if count_syllables(combined) <= max_allowed:
                merged.append(combined)
                i += 2
                continue
        elif syl < min_allowed and merged:
            prev_line = merged[-1]
            combined = f"{prev_line} {line}"
            if count_syllables(combined) <= max_allowed:
                merged[-1] = combined
                i += 1
                continue

        merged.append(line)
        i += 1

    return merged


def structure_lines(
    lines: list[str],
    min_allowed: int = 5,
    max_allowed: int = 15,
) -> list[str]:
    """초과 라인 분할 및 단문 병합을 통합 수행하여 안정적인 음절 수 라인 리스트를 반환합니다."""
    structured = []
    for i, line in enumerate(lines):
        prev_l = structured[-1] if structured else ""
        next_l = lines[i + 1] if i + 1 < len(lines) else ""
        sub_lines = split_line_by_rhyme(
            line,
            min_allowed=min_allowed,
            max_allowed=max_allowed,
            prev_line=prev_l,
            next_line=next_l,
        )
        structured.extend(sub_lines)

    return merge_underlength_lines(structured, min_allowed=min_allowed, max_allowed=max_allowed)
