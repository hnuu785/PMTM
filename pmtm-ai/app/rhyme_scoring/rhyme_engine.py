import re
from dataclasses import dataclass

try:
    from phonetics_utils import VOWEL_GROUPS, get_phonemes
except ImportError:
    from .phonetics_utils import VOWEL_GROUPS, get_phonemes

def calculate_syllable_score(s1, s2):
    """두 음절 간의 라이밍 점수 계산 (Vowel 100%)"""
    v_score = 0.0
    
    # 1. 모음 점수 (100%)
    if s1['v'] == s2['v']:
        v_score = 1.0
    elif any(s1['v'] in g and s2['v'] in g for g in VOWEL_GROUPS):
        v_score = 1.0  # 유사 모음 점수 (1.0으로 통일)
        
    return v_score

def _match_with_offset(p1: list[dict], p2: list[dict], offset1: int, offset2: int) -> float:
    sub_p1 = p1[:-offset1] if offset1 > 0 else p1
    sub_p2 = p2[:-offset2] if offset2 > 0 else p2
    
    min_len = min(len(sub_p1), len(sub_p2), 3)
    if min_len <= 0:
        return 0.0

    # 맨 끝 음절(앵커 음절) 불일치 시 라임 미성립 (0.0 반환)
    if calculate_syllable_score(sub_p1[-1], sub_p2[-1]) == 0.0:
        return 0.0
        
    total_score = 0.0
    for i in range(1, min_len + 1):
        s1 = sub_p1[-i]
        s2 = sub_p2[-i]
        score = calculate_syllable_score(s1, s2)
        
        weight = 1.0 if i == 1 else (0.5 if i == 2 else 0.3)
        total_score += score * weight
        
    max_possible = sum([1.0 if i == 1 else (0.5 if i == 2 else 0.3) for i in range(1, min_len + 1)])
    return total_score / max_possible if max_possible > 0 else 0.0


def _terminal_ending_group(word: str) -> str | None:
    """Return a grammatical ending family that should not count as a lyric rhyme."""
    if word.endswith(("습니다", "니다")):
        return "formal"
    if word.endswith(("요", "죠")):
        return "polite"
    if word.endswith("네"):
        return "exclamatory"
    return None


def get_line_rhyme_score(line1, line2):
    """두 문장 끝단어 간의 라임 점수 계산 (끝에서 최대 3음절, 우측 정렬 매칭)"""
    # 0) 동일 단어 또는 같은 문법 종결어미 반복은 라임으로 보지 않는다.
    w1 = line1.strip().split()[-1] if line1.strip().split() else ""
    w2 = line2.strip().split()[-1] if line2.strip().split() else ""

    w1_clean = re.sub(r"[^\w]", "", w1).lower()
    w2_clean = re.sub(r"[^\w]", "", w2).lower()

    if w1_clean == w2_clean and w1_clean != "":
        return 0.2  # 동일 단어 단순 반복 시 낮은 점수(0.2) 부여

    ending1 = _terminal_ending_group(w1_clean)
    ending2 = _terminal_ending_group(w2_clean)
    if ending1 is not None and ending1 == ending2:
        return 0.2

    p1 = get_phonemes(line1)
    p2 = get_phonemes(line2)
    
    if not p1 or not p2:
        return 0.0
    
    # 오프셋 0 (우측 정렬 매칭만 수행)
    return round(_match_with_offset(p1, p2, 0, 0), 4)


@dataclass(frozen=True)
class BarEndRhymeAnalysis:
    coverage_score: float
    selected_line_indexes: tuple[int, ...]
    line_scores: tuple[float, ...]
    rhyme_groups: tuple[int | None, ...]
    best_match_indexes: tuple[int | None, ...]


def check_end_rhyme_pair(line1: str, line2: str, threshold: float = 0.5) -> bool:
    """두 문장 끝단어 간의 라임 점수가 임계값 이상인지 검증합니다."""
    return get_line_rhyme_score(line1, line2) >= threshold


def _is_trap(bpm: float | None, genre: str | None) -> bool:
    if genre is not None:
        return genre.lower() in ("trap", "트랩")
    if bpm is None:
        return False
    judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
    return judgment_bpm >= 115


def get_bar_end_indexes(
    lines: list[str],
    bpm: float | None = None,
    genre: str | None = None,
) -> tuple[int, ...]:
    """Return all boombap line indexes or even trap line indexes."""
    if _is_trap(bpm, genre) and len(lines) >= 10:
        return tuple(range(1, len(lines), 2))
    return tuple(range(len(lines)))


def analyze_bar_end_rhyme(
    lines: list[str],
    bpm: float | None = None,
    genre: str | None = None,
    threshold: float = 0.5,
    # GRPO, inference, and dataset evaluation intentionally use adjacent endings only.
    # Presentation callers may override this, but must not change the shared scoring default.
    max_gap: int = 1,
) -> BarEndRhymeAnalysis:
    """Analyze bar-ending rhymes using the shared GRPO policy.

    A bar ending is covered when it rhymes with another bar ending within max_gap positions.
    Trap uses the second line of each two-line bar; boombap uses every line.
    Only covered bar endings are assigned groups and highlight matches.
    """
    line_count = len(lines)
    selected = get_bar_end_indexes(lines, bpm=bpm, genre=genre)
    line_scores = [0.0] * line_count
    rhyme_groups: list[int | None] = [None] * line_count
    best_match_indexes: list[int | None] = [None] * line_count
    if len(selected) < 2:
        return BarEndRhymeAnalysis(
            coverage_score=0.0,
            selected_line_indexes=selected,
            line_scores=tuple(line_scores),
            rhyme_groups=tuple(rhyme_groups),
            best_match_indexes=tuple(best_match_indexes),
        )

    parents = list(range(len(selected)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    gap_scores: dict[int, list[float]] = {}
    for gap in range(1, max_gap + 1):
        scores = []
        for position in range(len(selected) - gap):
            score = get_line_rhyme_score(lines[selected[position]], lines[selected[position + gap]])
            scores.append(score)
            if score >= threshold:
                union(position, position + gap)
        gap_scores[gap] = scores

    covered_positions = []
    for position, line_index in enumerate(selected):
        candidates: list[tuple[float, int, int]] = []
        for gap in range(1, max_gap + 1):
            scores = gap_scores[gap]
            if position >= gap:
                candidates.append((scores[position - gap], -gap, selected[position - gap]))
            if position < len(selected) - gap:
                candidates.append((scores[position], -gap, selected[position + gap]))
        if candidates:
            best_score, neg_gap, best_index = max(candidates, key=lambda c: (c[0], c[1]))
            if best_score >= threshold:
                covered_positions.append(position)
                line_scores[line_index] = best_score
                best_match_indexes[line_index] = best_index

    group_ids: dict[int, int] = {}
    for position in covered_positions:
        root = find(position)
        if root not in group_ids:
            group_ids[root] = len(group_ids)
        rhyme_groups[selected[position]] = group_ids[root]

    return BarEndRhymeAnalysis(
        coverage_score=round(len(covered_positions) / len(selected), 4),
        selected_line_indexes=selected,
        line_scores=tuple(line_scores),
        rhyme_groups=tuple(rhyme_groups),
        best_match_indexes=tuple(best_match_indexes),
    )


def calculate_chunk_end_rhyme_score(
    lines: list[str],
    bpm: float | None = None,
    genre: str | None = None,
) -> float:
    """Backward-compatible shared bar-ending rhyme coverage score."""
    return analyze_bar_end_rhyme(lines, bpm=bpm, genre=genre).coverage_score

def calculate_line_scores(
    actual_lines: list[str],
    bpm: float | None = None,
    genre: str | None = None,
) -> tuple[list[float], list[int | None]]:
    """
    각 라인별 연속 라임 점수와 최적 매칭 인접 라인 인덱스 계산.
    (AI 학습의 reward 채점 로직과 동일)

    bpm 또는 genre 매개변수를 기준으로 트랩(genre="트랩" 또는 bpm >= 115) 환경에서는
    건너뛴 라임(ABAC, BACA 등 i <-> i+2) 가중치를 1.3333 (line_score 기준 0.8점 반영)으로 상향 조절합니다.
    트랩 환경에서 10줄 이상(16줄 구조, 2줄 = 1마디)인 경우, 마디의 끝이 되는 짝수번째 라인(index 1, 3, 5...)을 각운으로 평가합니다.
    """
    actual_len = len(actual_lines)
    if actual_len <= 1:
        return [0.0] * actual_len, [None] * actual_len

    # 장르 판별 (트랩 여부 확인)
    is_trap = False
    if genre is not None:
        is_trap = genre.lower() in ("trap", "트랩")
    elif bpm is not None:
        judgment_bpm = bpm * 2.0 if 60.0 <= bpm < 80.0 else bpm
        is_trap = judgment_bpm >= 115

    # 트랩 장르이고 10줄 이상(예: 16줄 구조, 2줄 = 1마디)인 경우:
    # 짝수번째 라인(0-indexed 기준 index 1, 3, 5, 7...)만 추출하여 마디 각운으로 평가 (8마디 표준 채점과 동일)
    if is_trap and actual_len >= 10:
        even_indices = list(range(1, actual_len, 2))
        even_lines = [actual_lines[idx] for idx in even_indices]

        # 짝수번째 8개 라인은 이미 8마디 표준 각운들이므로 붐뱁(8마디 표준) 알고리즘으로 라인 스코어 산출
        sub_scores, sub_matches = calculate_line_scores(even_lines, bpm=bpm, genre="붐뱁")

        line_scores = [0.0] * actual_len
        best_match_indexes = [None] * actual_len

        for k, idx in enumerate(even_indices):
            line_scores[idx] = sub_scores[k]
            if idx - 1 >= 0:
                line_scores[idx - 1] = sub_scores[k]

            if sub_matches[k] is not None:
                best_match_indexes[idx] = even_indices[sub_matches[k]]
                if idx - 1 >= 0:
                    best_match_indexes[idx - 1] = even_indices[sub_matches[k]]

        return line_scores, best_match_indexes

    # 인접한 모든 쌍(i, i+1)의 라임 점수를 계산
    adj_scores = []
    for i in range(actual_len - 1):
        score = get_line_rhyme_score(actual_lines[i], actual_lines[i+1])
        adj_scores.append(score)

    # 한 줄 건너뛴 쌍(i, i+2)의 라임 점수를 계산
    skip_scores = []
    for i in range(actual_len - 2):
        score = get_line_rhyme_score(actual_lines[i], actual_lines[i+2])
        skip_scores.append(score)

    # 건너뛴 라임 가중치: 붐뱁 0.5 (줄당 0.3점), 트랩 1.3333 (줄당 0.8점)
    skip_factor = (4.0 / 3.0) if is_trap else 0.5

    line_scores = []
    best_match_indexes = [None] * actual_len

    for i in range(actual_len):
        # 2-1) AA 판정 (인접 및 한 줄 건너뛴 라임 결합)
        score_L1 = adj_scores[i-1] if i > 0 else 0.0
        score_R1 = adj_scores[i] if i < actual_len - 1 else 0.0

        score_L2 = skip_scores[i-2] if i > 1 else 0.0
        score_R2 = skip_scores[i] if i < actual_len - 2 else 0.0
        skip_max = max(score_L2, score_R2)

        # 인접 라인은 1.0 가중치, 건너뛴 라임은 skip_factor 가중치 적용하여 최댓값 선택
        adj_max = max(score_L1, score_R1, skip_factor * skip_max)

        # best_match_index 결정 (인접 라인 우선, 없으면 건너뛴 라인)
        if adj_max > 0.0:
            if score_L1 > 0.0 or score_R1 > 0.0:
                if score_L1 >= score_R1 and i > 0:
                    best_match_indexes[i] = i - 1
                elif score_R1 > score_L1 and i < actual_len - 1:
                    best_match_indexes[i] = i + 1
            elif skip_max > 0.0:
                if score_L2 >= score_R2 and i > 1:
                    best_match_indexes[i] = i - 2
                elif score_R2 > score_L2 and i < actual_len - 2:
                    best_match_indexes[i] = i + 2

        # 2-2) AAA 판정
        consec_3_cases = []
        if i >= 2:
            consec_3_cases.append(min(adj_scores[i-2], adj_scores[i-1]))
        if i >= 1 and i < actual_len - 1:
            consec_3_cases.append(min(adj_scores[i-1], adj_scores[i]))
        if i < actual_len - 2:
            consec_3_cases.append(min(adj_scores[i], adj_scores[i+1]))
        consec_3 = max(consec_3_cases) if consec_3_cases else 0.0

        # 2-3) AAAA 판정
        consec_4_cases = []
        if i >= 3:
            consec_4_cases.append(min(adj_scores[i-3], adj_scores[i-2], adj_scores[i-1]))
        if i >= 2 and i < actual_len - 1:
            consec_4_cases.append(min(adj_scores[i-2], adj_scores[i-1], adj_scores[i]))
        if i >= 1 and i < actual_len - 2:
            consec_4_cases.append(min(adj_scores[i-1], adj_scores[i], adj_scores[i+1]))
        if i < actual_len - 3:
            consec_4_cases.append(min(adj_scores[i], adj_scores[i+1], adj_scores[i+2]))
        consec_4 = max(consec_4_cases) if consec_4_cases else 0.0

        # 가중치 결합 (w_aa=0.6, w_aaa=0.2, w_aaaa=0.2)
        w_aa, w_aaa, w_aaaa = 0.6, 0.2, 0.2
        line_score = w_aa * adj_max + w_aaa * consec_3 + w_aaaa * consec_4
        line_scores.append(line_score)

    return line_scores, best_match_indexes

