try:
    from phonetics_utils import VOWEL_GROUPS, CODA_GROUPS, get_phonemes
except ImportError:
    from .phonetics_utils import VOWEL_GROUPS, CODA_GROUPS, get_phonemes

def calculate_syllable_score(s1, s2):
    """두 음절 간의 라이밍 점수 계산 (Vowel 80%, Coda 20%)"""
    v_score = 0.0
    c_score = 0.0
    
    # 1. 모음 점수 (80%)
    if s1['v'] == s2['v']:
        v_score = 1.0
    elif VOWEL_GROUPS.get(s1['v']) == s2['v']:
        v_score = 0.8  # 유사 모음 점수
        
    # 2. 종성(받침) 점수 (20%)
    if s1['c'] == s2['c']:
        c_score = 1.0
    elif s1['c'] and s2['c'] and CODA_GROUPS.get(s1['c']) == CODA_GROUPS.get(s2['c']):
        c_score = 0.7  # 유사 종성 점수
    elif not s1['c'] and not s2['c']:
        c_score = 1.0  # 둘 다 받침이 없는 경우 (청각적 일관성)
    elif (s1['c'] and not s2['c']) or (not s1['c'] and s2['c']):
        c_score = 0.0  # 받침 유무가 다른 경우 감점
        
    return (v_score * 0.8) + (c_score * 0.2)

def get_line_rhyme_score(line1, line2):
    """두 문장 끝단어 간의 라임 점수 계산 (끝에서 최대 3음절)"""
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
