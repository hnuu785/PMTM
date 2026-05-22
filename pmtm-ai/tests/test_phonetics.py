"""Phonetics 회귀 테스트 — rhyme_scoring 모듈 sanity check.

pytest 없이 직접 실행: python tests/test_phonetics.py
종료 코드: 0=성공, 1=실패 (run_training.py가 이를 보고 학습 중단)

g2pk가 설치되어 있어도/없어도 통과하도록 테스트 케이스는 연음/구개음화를
거치지 않는 단순 음절만 사용한다.
"""

import io
import sys
from pathlib import Path

# Windows cp949 콘솔에서도 한글/유니코드 출력 가능하도록
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rhyme_scoring.phonetics_utils import CODA_GROUPS, VOWEL_GROUPS, get_phonemes
from app.rhyme_scoring.rhyme_engine import calculate_syllable_score, get_line_rhyme_score


failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    msg = f"[{tag}] {name}"
    if not cond and detail:
        msg += f"  ->  {detail}"
    print(msg)
    if not cond:
        failures.append(name)


def approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol


# ─── 1. 한글 음소 분해 ────────────────────────────────────
print("=== 한글 음소 분해 ===")

p = get_phonemes("강")
check("'강' -> 1 음절", len(p) == 1, f"got {p}")
check("'강' = {ㅏ, ㅇ}",
      len(p) == 1 and p[0] == {"v": "ㅏ", "c": "ㅇ"}, f"got {p}")

p = get_phonemes("바다")
check("'바다' -> 2 음절", len(p) == 2, f"got {p}")
check("'바다' 종성 모두 없음",
      len(p) == 2 and p[0]["c"] is None and p[1]["c"] is None, f"got {p}")

p = get_phonemes("한국어")
check("'한국어' -> 3 음절", len(p) == 3, f"got {p}")

# dict 구조 보장
p = get_phonemes("랩")
check("phoneme dict 키 = {v, c}",
      all(set(s.keys()) == {"v", "c"} for s in p), f"got {p}")


# ─── 2. 영어 음소 분해 (CMU 또는 글자 폴백) ───────────────
print("\n=== 영어 음소 분해 ===")

p = get_phonemes("rap")
check("'rap' ≥ 1 음절", len(p) >= 1, f"got {p}")

p = get_phonemes("hello")
check("'hello' ≥ 1 음절", len(p) >= 1, f"got {p}")

# 한영 혼합
p = get_phonemes("hello 강")
check("한영 혼합 토큰 처리 가능", len(p) >= 2, f"got {p}")


# ─── 3. 엣지 케이스 ──────────────────────────────────────
print("\n=== 엣지 케이스 ===")

check("빈 문자열 -> []", get_phonemes("") == [])
check("숫자만 -> []", get_phonemes("12345") == [])
check("특수문자만 -> []", get_phonemes("!@#$%") == [])


# ─── 4. 음절 라임 점수 (가중치 0.8*v + 0.2*c) ─────────────
print("\n=== 음절 라임 점수 ===")

gang = get_phonemes("강")[0]   # {ㅏ, ㅇ}
bang = get_phonemes("방")[0]   # {ㅏ, ㅇ}
san = get_phonemes("산")[0]    # {ㅏ, ㄴ}
ga = get_phonemes("가")[0]     # {ㅏ, None}
bom = get_phonemes("봄")[0]    # {ㅗ, ㅁ}

s = calculate_syllable_score(gang, gang)
check("자기 자신 = 1.0", approx(s, 1.0), f"got {s}")

s = calculate_syllable_score(gang, bang)
check("강↔방 (둘다 ㅏ+ㅇ) = 1.0", approx(s, 1.0), f"got {s}")

s = calculate_syllable_score(gang, san)
# v=1 (ㅏ=ㅏ), c=0.7 (ㅇ,ㄴ 둘 다 비음) -> 0.8 + 0.14 = 0.94
check("강↔산 (모음 동일, 종성 유사 비음) ~= 0.94",
      approx(s, 0.94, 0.01), f"got {s}")

s = calculate_syllable_score(gang, ga)
# v=1 (ㅏ=ㅏ), c=0 (ㅇ vs None -> 한쪽만 있음) -> 0.8 + 0 = 0.8
check("강↔가 (종성 유무 불일치) ~= 0.8",
      approx(s, 0.8, 0.01), f"got {s}")

s = calculate_syllable_score(gang, bom)
# v=0 (ㅏ↔ㅗ 그룹 아님), c=0.7 (ㅇ,ㅁ 비음) -> 0 + 0.14 = 0.14
check("강↔봄 (모음 다름, 종성 유사) ~= 0.14",
      approx(s, 0.14, 0.01), f"got {s}")


# ─── 5. 줄 단위 라임 점수 ─────────────────────────────────
print("\n=== 줄 단위 라임 점수 ===")

s = get_line_rhyme_score("강", "방")
check("'강'↔'방' = 1.0", approx(s, 1.0), f"got {s}")

s = get_line_rhyme_score("강물", "방물")
# 끝 2음절 모두 동일 -> 1.0
check("'강물'↔'방물' = 1.0", approx(s, 1.0), f"got {s}")

s = get_line_rhyme_score("바다", "마차")
# 바(ㅏ,_)다(ㅏ,_) vs 마(ㅏ,_)차(ㅏ,_) -> 모든 끝 음절 일치 -> 1.0
check("'바다'↔'마차' = 1.0", approx(s, 1.0), f"got {s}")

s = get_line_rhyme_score("", "강")
check("빈 줄 (좌) = 0.0", s == 0.0, f"got {s}")

s = get_line_rhyme_score("강", "")
check("빈 줄 (우) = 0.0", s == 0.0, f"got {s}")

# 점수는 [0, 1] 구간 (회귀 가드)
for a, b in [("내일 가자", "오늘 가자"),
             ("랩 한다", "잽 든다"),
             ("hello", "world")]:
    s = get_line_rhyme_score(a, b)
    check(f"'{a}' ↔ '{b}' in [0,1]", 0.0 <= s <= 1.0, f"got {s}")


# ─── 6. 그룹 정의 sanity ─────────────────────────────────
print("\n=== 그룹 정의 ===")
check("VOWEL_GROUPS 비어있지 않음", len(VOWEL_GROUPS) > 0)
check("CODA_GROUPS 비어있지 않음", len(CODA_GROUPS) > 0)
check("CODA_GROUPS 값은 한국어 조음군 문자열",
      all(v in {"비음", "폐쇄", "유음"} for v in CODA_GROUPS.values()))


# ─── 결과 ────────────────────────────────────────────────
print()
if failures:
    print(f"FAILED: {len(failures)}개")
    for name in failures:
        print(f"  - {name}")
    sys.exit(1)

print("모든 테스트 통과 OK")
sys.exit(0)
