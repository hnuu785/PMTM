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

def get_line_rhyme_score(line1, line2):
    """두 문장 끝단어 간의 라임 점수 계산 (끝에서 최대 3음절)"""
    # 0) 어미/동일 단어 단순 반복 감점: 마지막 어절이 철자법상 완전히 동일한 경우 0점 처리
    w1 = line1.strip().split()[-1] if line1.strip().split() else ""
    w2 = line2.strip().split()[-1] if line2.strip().split() else ""
    
    import re
    w1_clean = re.sub(r"[^\w]", "", w1).lower()
    w2_clean = re.sub(r"[^\w]", "", w2).lower()
    
    if w1_clean == w2_clean and w1_clean != "":
        return 0.2  # 동일 단어 단순 반복 시 낮은 점수(0.2) 부여

    p1 = get_phonemes(line1)
    p2 = get_phonemes(line2)
    
    if not p1 or not p2:
        return 0.0
    
    # 끝에서부터 비교
    min_len = min(len(p1), len(p2), 3)
    total_score = 0.0
    
    for i in range(1, min_len + 1):
        s1 = p1[-i]
        s2 = p2[-i]
        score = calculate_syllable_score(s1, s2)
        
        # 가중치: 가장 끝 음절(1번째)이 가장 중요함
        weight = 1.0 if i == 1 else (0.5 if i == 2 else 0.3)
        total_score += score * weight
        
    # 정규화
    max_possible = sum([1.0 if i == 1 else (0.5 if i == 2 else 0.3) for i in range(1, min_len + 1)])
    
    return round(total_score / max_possible, 4) if max_possible > 0 else 0.0

def calculate_line_scores(actual_lines: list[str]) -> tuple[list[float], list[int | None]]:
    """
    각 라인별 연속 라임 점수와 최적 매칭 인접 라인 인덱스 계산.
    (AI 학습의 reward 채점 로직과 동일)
    """
    actual_len = len(actual_lines)
    if actual_len <= 1:
        return [0.0] * actual_len, [None] * actual_len

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

    line_scores = []
    best_match_indexes = [None] * actual_len

    for i in range(actual_len):
        # 2-1) AA 판정 (인접 및 한 줄 건너뛴 라임 결합)
        score_L1 = adj_scores[i-1] if i > 0 else 0.0
        score_R1 = adj_scores[i] if i < actual_len - 1 else 0.0

        score_L2 = skip_scores[i-2] if i > 1 else 0.0
        score_R2 = skip_scores[i] if i < actual_len - 2 else 0.0
        skip_max = max(score_L2, score_R2)

        # 인접 라인은 1.0 가중치, 건너뛴 라임은 0.5 가중치 적용하여 최댓값 선택
        adj_max = max(score_L1, score_R1, 0.5 * skip_max)

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


def calculate_rhyme_density(lines: list[str]) -> float:
    """마디 리스트 전체의 평균 복합 라임 점수를 산출합니다."""
    actual_len = len(lines)
    if actual_len <= 1:
        return 0.0
    line_scores, _ = calculate_line_scores(lines)
    return sum(line_scores) / actual_len
