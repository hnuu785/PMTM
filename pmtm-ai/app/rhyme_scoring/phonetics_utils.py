import re
from functools import lru_cache
from jamo import hangul_to_jamo, j2hcj

try:
    from g2pk import G2p
    _g2p_kr_inst = G2p()

    @lru_cache(maxsize=20000)
    def _g2p_kr(text: str) -> str:
        return _g2p_kr_inst(text)
except Exception:
    _g2p_kr = None

try:
    import pronouncing
except ImportError:
    pronouncing = None


# 유사 모음 그룹 (현대 한국어 발음의 유사성 기준)
VOWEL_GROUPS = {
    'ㅐ': 'ㅔ', 'ㅔ': 'ㅐ', 'ㅖ': 'ㅔ', 'ㅒ': 'ㅐ',
    'ㅗ': 'ㅜ', 'ㅜ': 'ㅗ',
    'ㅡ': 'ㅣ', 'ㅣ': 'ㅡ',
    'ㅏ': 'ㅑ', 'ㅑ': 'ㅏ',
    'ㅓ': 'ㅕ', 'ㅕ': 'ㅓ',
    'ㅙ': 'ㅐ', 'ㅚ': 'ㅔ', 'ㅞ': 'ㅔ',
}

# 종성(받침) 조음 그룹
CODA_GROUPS = {
    'ㄴ': '비음', 'ㅁ': '비음', 'ㅇ': '비음',
    'ㄱ': '폐쇄', 'ㅂ': '폐쇄', 'ㄷ': '폐쇄', 'ㅅ': '폐쇄',
    'ㅋ': '폐쇄', 'ㅌ': '폐쇄', 'ㅍ': '폐쇄',
    'ㄹ': '유음',
}

# CMU vowel → 한글 중성 (단순화: 강세/이중모음 일부 손실 감수)
CMU_VOWEL_TO_JAMO = {
    'AA': 'ㅏ', 'AE': 'ㅐ', 'AH': 'ㅓ', 'AO': 'ㅗ',
    'AW': 'ㅏ', 'AY': 'ㅏ',
    'EH': 'ㅔ', 'ER': 'ㅓ', 'EY': 'ㅔ',
    'IH': 'ㅣ', 'IY': 'ㅣ',
    'OW': 'ㅗ', 'OY': 'ㅗ',
    'UH': 'ㅜ', 'UW': 'ㅜ',
}

# CMU consonant → 종성 jamo (한국어 7종성 체계 근사)
CMU_CONS_TO_CODA = {
    'B': 'ㅂ', 'P': 'ㅂ', 'F': 'ㅂ', 'V': 'ㅂ',
    'D': 'ㄷ', 'T': 'ㄷ', 'TH': 'ㄷ', 'DH': 'ㄷ',
    'G': 'ㄱ', 'K': 'ㄱ',
    'M': 'ㅁ', 'N': 'ㄴ', 'NG': 'ㅇ',
    'L': 'ㄹ', 'R': 'ㄹ',
    'S': 'ㅅ', 'Z': 'ㅅ', 'SH': 'ㅅ', 'ZH': 'ㅅ',
    'CH': 'ㅅ', 'JH': 'ㅅ',
    'HH': None, 'W': None, 'Y': None,
}

# fallback (CMU 사전에 없는 단어): 자모 직접 대응
_LETTER_V = {'a': 'ㅏ', 'e': 'ㅔ', 'i': 'ㅣ', 'o': 'ㅗ', 'u': 'ㅜ', 'y': 'ㅣ'}
_LETTER_C = {
    'b': 'ㅂ', 'c': 'ㄱ', 'd': 'ㄷ', 'f': 'ㅂ', 'g': 'ㄱ',
    'j': 'ㅅ', 'k': 'ㄱ', 'l': 'ㄹ', 'm': 'ㅁ', 'n': 'ㄴ',
    'p': 'ㅂ', 'q': 'ㄱ', 'r': 'ㄹ', 's': 'ㅅ', 't': 'ㄷ',
    'v': 'ㅂ', 'x': 'ㄱ', 'z': 'ㅅ', 'h': None, 'w': None,
}


def _strip_stress(p: str) -> str:
    return re.sub(r'\d', '', p)


def _cmu_to_phonemes(cmu_str: str) -> list[dict]:
    """CMU phones → [{v, c}, ...] (음절 단위)"""
    phones = [_strip_stress(p) for p in cmu_str.split()]
    vowel_idxs = [i for i, p in enumerate(phones) if p in CMU_VOWEL_TO_JAMO]
    if not vowel_idxs:
        return []
    out = []
    for k, vi in enumerate(vowel_idxs):
        v = CMU_VOWEL_TO_JAMO[phones[vi]]
        if k + 1 < len(vowel_idxs):
            # 다음 모음까지 사이의 자음 — 마지막 자음은 다음 음절의 onset, 그 앞은 coda
            between = phones[vi + 1:vowel_idxs[k + 1]]
            coda = CMU_CONS_TO_CODA.get(between[0]) if len(between) >= 2 else None
        else:
            # 마지막 모음 — 뒤따르는 자음 모두 coda 후보, 첫 자음을 coda로
            trailing = phones[vi + 1:]
            coda = CMU_CONS_TO_CODA.get(trailing[0]) if trailing else None
        out.append({'v': v, 'c': coda})
    return out


def _english_fallback(word: str) -> list[dict]:
    """CMU 사전 미등재 단어용 — 글자 단위 근사."""
    word = word.lower()
    out = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in _LETTER_V:
            # 다음 모음 위치 탐색
            j = i + 1
            while j < len(word) and word[j] not in _LETTER_V:
                j += 1
            cons_run = word[i + 1:j]
            if not cons_run:
                coda = None
            elif j == len(word) or len(cons_run) >= 2:
                coda = _LETTER_C.get(cons_run[0])
            else:
                # 단자음이 다음 모음의 onset이 됨
                coda = None
            out.append({'v': _LETTER_V[ch], 'c': coda})
            i = j
        else:
            i += 1
    return out


def _english_word_to_phonemes(word: str) -> list[dict]:
    if pronouncing is not None:
        cands = pronouncing.phones_for_word(word.lower())
        if cands:
            return _cmu_to_phonemes(cands[0])
    return _english_fallback(word)


def _hangul_to_phonemes(text: str) -> list[dict]:
    out = []
    for ch in text:
        if not ('가' <= ch <= '힣'):
            continue
        try:
            jamo_str = j2hcj(hangul_to_jamo(ch))
            v = jamo_str[1]
            c = jamo_str[2] if len(jamo_str) > 2 else None
            out.append({'v': v, 'c': c})
        except Exception:
            continue
    return out


def normalize_pronunciation(text: str) -> list[dict]:
    """텍스트를 한글/영어 토큰으로 분기해 발음 처리 후 [{v,c}, ...]로 통합."""
    tokens = re.findall(r'[가-힣]+|[A-Za-z]+', text)
    out = []
    for tok in tokens:
        if '가' <= tok[0] <= '힣':
            if _g2p_kr is not None:
                try:
                    tok = _g2p_kr(tok)
                except Exception:
                    pass
            out.extend(_hangul_to_phonemes(tok))
        else:
            out.extend(_english_word_to_phonemes(tok))
    return out


def get_phonemes(text: str) -> list[dict]:
    """기존 인터페이스 유지 — rhyme_engine에서 import."""
    return normalize_pronunciation(text)
