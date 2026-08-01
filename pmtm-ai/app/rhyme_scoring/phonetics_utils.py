import re
from functools import lru_cache
# Use mathematical index decomposition to avoid jamo package dependency.

try:
    from loanword_overrides import MANUAL_LOANWORD_OVERRIDES
except ImportError:
    from .loanword_overrides import MANUAL_LOANWORD_OVERRIDES



try:
    # pyrefly: ignore [missing-import]
    from g2pk import G2p
    _g2p_kr_inst = G2p()

    @lru_cache(maxsize=20000)
    def _g2p_kr(text: str) -> str:
        return _g2p_kr_inst(text)
except Exception:
    _g2p_kr = None

try:
    # pyrefly: ignore [missing-import]
    import pronouncing
except ImportError:
    pronouncing = None


# 유사 모음 군집 (현대 한국어 발음의 유사성 기준)
VOWEL_GROUPS = [
    {'ㅐ', 'ㅔ', 'ㅖ', 'ㅒ', 'ㅙ', 'ㅚ', 'ㅞ'},  # 에/애/예/얘/왜/외/웨 계열 통합
    {'ㅏ', 'ㅑ', 'ㅘ'},
    {'ㅓ', 'ㅕ', 'ㅝ'},
    {'ㅗ', 'ㅛ'},
    {'ㅡ', 'ㅜ'},
    {'ㅜ', 'ㅠ'}
]

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
_CONS_DIGRAPHS = {'th', 'sh', 'ch', 'wh', 'ph', 'gh', 'ck', 'qu', 'ng'}
_FINAL_STOP_LETTERS = set('bdgkpt')
_LETTER_V = {'a': 'ㅏ', 'e': 'ㅔ', 'i': 'ㅣ', 'o': 'ㅗ', 'u': 'ㅜ', 'y': 'ㅣ'}
_LETTER_C = {
    'b': 'ㅂ', 'c': 'ㄱ', 'd': 'ㄷ', 'f': 'ㅂ', 'g': 'ㄱ',
    'j': 'ㅅ', 'k': 'ㄱ', 'l': 'ㄹ', 'm': 'ㅁ', 'n': 'ㄴ',
    'p': 'ㅂ', 'q': 'ㄱ', 'r': 'ㄹ', 's': 'ㅅ', 't': 'ㄷ',
    'v': 'ㅂ', 'x': 'ㄱ', 'z': 'ㅅ', 'h': None, 'w': None,
}

# 알파벳을 한 글자씩 읽는 약어용 표. (OG → 오지, MC → 엠씨)
_LETTER_NAME_KR = {
    'A': '에이', 'B': '비', 'C': '씨', 'D': '디', 'E': '이', 'F': '에프',
    'G': '지', 'H': '에이치', 'I': '아이', 'J': '제이', 'K': '케이', 'L': '엘',
    'M': '엠', 'N': '엔', 'O': '오', 'P': '피', 'Q': '큐', 'R': '알',
    'S': '에스', 'T': '티', 'U': '유', 'V': '브이', 'W': '더블유', 'X': '엑스',
    'Y': '와이', 'Z': '지',
}

_BUILTIN_LOANWORD_OVERRIDES = {
    "apple": "애플",
    "orange": "오렌지",
    "party": "파티",
    "money": "머니",
    "vibe": "바이브",
    "radio": "라디오",
    "cookie": "쿠키",
}

_LOANWORD_OVERRIDES = {
    **_BUILTIN_LOANWORD_OVERRIDES,
    **MANUAL_LOANWORD_OVERRIDES,
}

_CHOSEONG_INDEX = {
    None: 11,
    'ㄱ': 0, 'ㄲ': 1, 'ㄴ': 2, 'ㄷ': 3, 'ㄸ': 4, 'ㄹ': 5, 'ㅁ': 6, 'ㅂ': 7, 'ㅃ': 8,
    'ㅅ': 9, 'ㅆ': 10, 'ㅇ': 11, 'ㅈ': 12, 'ㅉ': 13, 'ㅊ': 14, 'ㅋ': 15, 'ㅌ': 16,
    'ㅍ': 17, 'ㅎ': 18,
}

_JUNGSEONG_INDEX = {
    'ㅏ': 0, 'ㅐ': 1, 'ㅑ': 2, 'ㅒ': 3, 'ㅓ': 4, 'ㅔ': 5, 'ㅕ': 6, 'ㅖ': 7, 'ㅗ': 8,
    'ㅘ': 9, 'ㅙ': 10, 'ㅚ': 11, 'ㅛ': 12, 'ㅜ': 13, 'ㅝ': 14, 'ㅞ': 15, 'ㅟ': 16,
    'ㅠ': 17, 'ㅡ': 18, 'ㅢ': 19, 'ㅣ': 20,
}

_JONGSEONG_INDEX = {
    None: 0,
    'ㄱ': 1, 'ㄲ': 2, 'ㄳ': 3, 'ㄴ': 4, 'ㄵ': 5, 'ㄶ': 6, 'ㄷ': 7, 'ㄹ': 8, 'ㄺ': 9,
    'ㄻ': 10, 'ㄼ': 11, 'ㄽ': 12, 'ㄾ': 13, 'ㄿ': 14, 'ㅀ': 15, 'ㅁ': 16, 'ㅂ': 17,
    'ㅄ': 18, 'ㅅ': 19, 'ㅆ': 20, 'ㅇ': 21, 'ㅈ': 22, 'ㅊ': 23, 'ㅋ': 24, 'ㅌ': 25,
    'ㅍ': 26, 'ㅎ': 27,
}

_CMU_CONS_TO_ONSET = {
    'B': 'ㅂ', 'P': 'ㅍ', 'F': 'ㅍ', 'V': 'ㅂ',
    'D': 'ㄷ', 'T': 'ㅌ', 'TH': 'ㅌ', 'DH': 'ㄷ',
    'G': 'ㄱ', 'K': 'ㅋ',
    'M': 'ㅁ', 'N': 'ㄴ', 'NG': 'ㅇ',
    'L': 'ㄹ', 'R': 'ㄹ',
    'S': 'ㅅ', 'Z': 'ㅈ', 'SH': 'ㅅ', 'ZH': 'ㅈ',
    'CH': 'ㅊ', 'JH': 'ㅈ',
    'HH': 'ㅎ', 'W': 'ㅇ', 'Y': 'ㅇ',
}

_CMU_CONS_TO_CODA_LOAN = {
    'B': 'ㅂ', 'P': 'ㅂ', 'F': 'ㅂ', 'V': 'ㅂ',
    'D': 'ㄷ', 'T': 'ㄷ', 'TH': 'ㄷ', 'DH': 'ㄷ',
    'G': 'ㄱ', 'K': 'ㄱ',
    'M': 'ㅁ', 'N': 'ㄴ', 'NG': 'ㅇ',
    'L': 'ㄹ', 'R': 'ㄹ',
    'S': 'ㅅ', 'Z': 'ㅅ', 'SH': 'ㅅ', 'ZH': 'ㅅ',
    'CH': 'ㅊ', 'JH': 'ㅈ',
}

# 이중모음의 활음은 한국어에서 별도 음절이 된다. (grind → 그라인드, down → 다운)
# OW는 관용 표기가 갈려서(go → 고, flow → 플로우) 분해하지 않고 사전 항목에 맡긴다.
_DIPHTHONG_OFFGLIDE = {'AY': 'ㅣ', 'AW': 'ㅜ', 'OY': 'ㅣ', 'EY': 'ㅣ'}

# 어말 파열음. 장모음·이중모음 뒤에서는 '으'가 붙어 한 음절이 된다.
_LONG_VOWELS = {'IY', 'UW', 'EY', 'OW', 'AY', 'AW', 'OY'}
_FINAL_STOPS = {'P', 'T', 'K', 'B', 'G', 'D'}

_EPENTHETIC_VOWEL_BY_CONS = {
    'CH': 'ㅣ', 'JH': 'ㅣ', 'Y': 'ㅣ',
    'SH': 'ㅡ', 'ZH': 'ㅡ', 'S': 'ㅡ', 'Z': 'ㅡ',
    'L': 'ㅡ', 'R': 'ㅡ',
}


def _strip_stress(p: str) -> str:
    return re.sub(r'\d', '', p)


def _compose_hangul(onset: str | None, vowel: str, coda: str | None = None) -> str:
    onset_idx = _CHOSEONG_INDEX.get(onset, _CHOSEONG_INDEX[None])
    vowel_idx = _JUNGSEONG_INDEX[vowel]
    coda_idx = _JONGSEONG_INDEX.get(coda, 0)
    codepoint = 0xAC00 + (onset_idx * 21 + vowel_idx) * 28 + coda_idx
    return chr(codepoint)


def _loanword_vowel(phone: str, trailing: list[str]) -> str | None:
    if phone == 'AH' and trailing == ['L']:
        return 'ㅡ'
    return CMU_VOWEL_TO_JAMO.get(phone)


def _epenthetic_vowel(phone: str) -> str:
    return _EPENTHETIC_VOWEL_BY_CONS.get(phone, 'ㅡ')


def _append_epenthetic_syllables(phones: list[str]) -> str:
    out = []
    for phone in phones:
        onset = _CMU_CONS_TO_ONSET.get(phone)
        if onset is None:
            continue
        out.append(_compose_hangul(onset, _epenthetic_vowel(phone)))
    return ''.join(out)


def _cmu_to_loanword(cmu_str: str) -> str | None:
    phones = [_strip_stress(p) for p in cmu_str.split()]
    vowel_idxs = [i for i, p in enumerate(phones) if p in CMU_VOWEL_TO_JAMO]
    if not vowel_idxs:
        return None

    syllables: list[str] = []
    carry_onset = None
    if vowel_idxs[0] > 0:
        # 어두 자음군은 한국어에서 자음마다 삽입모음이 붙어 음절이 늘어난다.
        # 마지막 자음만 첫 모음의 초성이 되고 앞선 자음들은 각각 한 음절이 된다.
        # (street → 스트리트, crew → 크루). W, Y는 초성이 되어 이중모음을 이룬다.
        onset_cluster = phones[:vowel_idxs[0]]
        syllables.append(_append_epenthetic_syllables(onset_cluster[:-1]))
        carry_onset = _CMU_CONS_TO_ONSET.get(onset_cluster[-1])

    for k, vi in enumerate(vowel_idxs):
        next_vi = vowel_idxs[k + 1] if k + 1 < len(vowel_idxs) else None
        trailing = phones[vi + 1:next_vi] if next_vi is not None else phones[vi + 1:]
        vowel = _loanword_vowel(phones[vi], trailing)
        if vowel is None:
            continue

        onset = carry_onset
        coda = None
        carry_onset = None
        extra_tail: list[str] = []

        if next_vi is not None:
            if len(trailing) >= 2:
                coda = _CMU_CONS_TO_CODA_LOAN.get(trailing[0])
                carry_onset = _CMU_CONS_TO_ONSET.get(trailing[-1])
            elif len(trailing) == 1:
                carry_onset = _CMU_CONS_TO_ONSET.get(trailing[0])
        elif trailing:
            head = trailing[0]
            if head == 'D' or (head in _FINAL_STOPS and phones[vi] in _LONG_VOWELS):
                # 어말 파열음이 '으'를 얻어 한 음절이 되는 경우. (street → 스트리트, bad → 배드)
                # 단모음 뒤 무성 파열음은 받침으로 남는다. (cat → 캣, stop → 스탑)
                extra_tail = trailing
            else:
                coda = _CMU_CONS_TO_CODA_LOAN.get(head)
                extra_tail = trailing[1:]

        offglide = _DIPHTHONG_OFFGLIDE.get(phones[vi])
        if offglide is not None:
            syllables.append(_compose_hangul(onset, vowel, None))
            onset = None
            vowel = offglide

        syllables.append(_compose_hangul(onset, vowel, coda))
        if extra_tail:
            syllables.append(_append_epenthetic_syllables(extra_tail))

    return ''.join(syllables) or None


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


def _acronym_to_phonemes(word: str) -> list[dict] | None:
    """대문자 약어를 글자 이름으로 읽는다. (OG → 오지, LA → 엘에이)

    네 글자 이상은 GRAY, MKIT처럼 대문자로 표기한 이름·단어가 섞여 있어 제외한다.
    """
    if not (word.isupper() and 2 <= len(word) <= 3):
        return None
    if not all(ch in _LETTER_NAME_KR for ch in word):
        return None
    return _hangul_to_phonemes(''.join(_LETTER_NAME_KR[ch] for ch in word))


def _digits_to_phonemes(token: str) -> list[dict]:
    """숫자를 한국어 수사로 읽는다. count_syllables와 같은 규칙을 쓴다."""
    try:
        return _hangul_to_phonemes(int_to_korean(int(token)))
    except Exception:
        return []


def _split_consonant_units(chunk: str) -> list[str]:
    """자음 나열을 발음 단위로 자른다. th, sh처럼 두 글자가 한 소리인 것은 묶는다."""
    units = []
    i = 0
    while i < len(chunk):
        if chunk[i:i + 2] in _CONS_DIGRAPHS:
            units.append(chunk[i:i + 2])
            i += 2
        else:
            units.append(chunk[i])
            i += 1
    return units


def _english_fallback(word: str) -> list[dict]:
    """CMU 사전 미등재 단어용 — 글자 단위 근사."""
    word = word.lower()
    out: list[dict] = []
    first_vowel = next((k for k, ch in enumerate(word) if ch in _LETTER_V), None)
    if first_vowel is None:
        return out

    # 어두 자음군은 마지막 단위만 초성이 되고 앞의 단위들은 'ㅡ' 음절이 된다.
    # (swag → 스왜그) 'th' 같은 두 글자 자음은 하나로 세어 음절이 늘지 않게 한다.
    for unit in _split_consonant_units(word[:first_vowel])[:-1]:
        if _LETTER_C.get(unit[0]) is not None:
            out.append({'v': 'ㅡ', 'c': None})

    i = first_vowel
    while i < len(word):
        ch = word[i]
        if ch not in _LETTER_V:
            i += 1
            continue

        # 연속된 모음 글자는 하나의 모음으로 본다. (feelin → 필린, homie → 호미)
        k = i
        while k + 1 < len(word) and word[k + 1] in _LETTER_V:
            k += 1
        j = k + 1
        while j < len(word) and word[j] not in _LETTER_V:
            j += 1

        cons_run = word[k + 1:j]
        coda = None
        tail_syllable = False
        if not cons_run:
            pass
        elif j == len(word):
            if cons_run not in _CONS_DIGRAPHS and cons_run[-1] in _FINAL_STOP_LETTERS:
                # 어말 파열음은 'ㅡ'가 붙어 한 음절이 된다. (swag → 스왜그)
                coda = _LETTER_C.get(cons_run[0]) if len(cons_run) >= 2 else None
                tail_syllable = True
            else:
                coda = _LETTER_C.get(cons_run[0])
        elif len(cons_run) >= 2:
            # 단자음은 다음 모음의 onset이 되므로 받침으로 잡지 않는다.
            coda = _LETTER_C.get(cons_run[0])

        out.append({'v': _LETTER_V[ch], 'c': coda})
        if tail_syllable:
            out.append({'v': 'ㅡ', 'c': None})
        i = j

    return out


def _english_word_to_phonemes(word: str) -> list[dict]:

    override = _LOANWORD_OVERRIDES.get(word.lower())
    if override is not None:
        return _hangul_to_phonemes(override)

    acronym = _acronym_to_phonemes(word)
    if acronym:
        return acronym

    if pronouncing is not None:
        cands = pronouncing.phones_for_word(word.lower())
        if cands:
            loanword = _cmu_to_loanword(cands[0])
            if loanword is not None:
                return _hangul_to_phonemes(loanword)
            return _cmu_to_phonemes(cands[0])
    return _english_fallback(word)


JUNGSEONG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
JONGSEONG = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ",
    "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ",
    "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)


def _hangul_to_phonemes(text: str) -> list[dict]:
    out = []
    for ch in text:
        if not ('가' <= ch <= '힣'):
            continue
        try:
            code = ord(ch) - 0xAC00
            v = JUNGSEONG[(code % 588) // 28]
            coda = JONGSEONG[code % 28]
            c = coda if coda else None
            out.append({'v': v, 'c': c})
        except Exception:
            continue
    return out


def normalize_pronunciation(text: str) -> list[dict]:
    """텍스트를 한글/영어 토큰으로 분기해 발음 처리 후 [{v,c}, ...]로 통합."""
    tokens = re.findall(r'[가-힣]+|[A-Za-z]+|\d+', text)
    out = []
    for tok in tokens:
        if '가' <= tok[0] <= '힣':
            if _g2p_kr is not None:
                try:
                    tok = _g2p_kr(tok)
                except Exception:
                    pass
            out.extend(_hangul_to_phonemes(tok))
        elif tok.isdigit():
            out.extend(_digits_to_phonemes(tok))
        else:
            out.extend(_english_word_to_phonemes(tok))
    return out


def get_phonemes(text: str) -> list[dict]:
    """기존 인터페이스 유지 — rhyme_engine에서 import."""
    return normalize_pronunciation(text)


def int_to_korean(n: int) -> str:
    if n == 0:
        return "영"
    units = ["", "십", "백", "천"]
    big_units = ["", "만", "억", "조"]
    digits = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    
    num_str = str(n)
    parts = []
    rev_str = num_str[::-1]
    for i in range(0, len(rev_str), 4):
        chunk = rev_str[i:i+4]
        chunk_val = ""
        for j, digit in enumerate(chunk):
            d = int(digit)
            if d > 0:
                if d == 1 and j > 0:
                    digit_name = ""
                else:
                    digit_name = digits[d]
                chunk_val = digit_name + units[j] + chunk_val
        if chunk_val:
            parts.append(chunk_val + big_units[i // 4])
    
    return "".join(reversed(parts))


def count_syllables(line: str) -> int:
    tokens = re.findall(r"[가-힣]+|[A-Za-z]+|\d+", line)
    total_syllables = 0
    for tok in tokens:
        if "가" <= tok[0] <= "힣":
            total_syllables += len(tok)
        elif tok.isdigit():
            try:
                korean_num = int_to_korean(int(tok))
                total_syllables += len(korean_num)
            except Exception:
                total_syllables += len(tok)
        else:
            try:
                phonemes = get_phonemes(tok)
                if phonemes:
                    total_syllables += len(phonemes)
                else:
                    vowels = re.findall(r"[aeiouyAEIOUY]", tok)
                    total_syllables += max(1, len(vowels))
            except Exception:
                vowels = re.findall(r"[aeiouyAEIOUY]", tok)
                total_syllables += max(1, len(vowels))
    return total_syllables
