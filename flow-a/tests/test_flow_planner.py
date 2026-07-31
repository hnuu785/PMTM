import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flow_planner import build_flow_plan, hangul_to_phonemes, to_diffsinger_ds


def make_analysis(bpm: int = 120, bars: int = 8) -> dict:
    slot_sec = 60 / bpm / 4
    grid = []
    for bar in range(bars):
        for slot in range(16):
            absolute_slot = bar * 16 + slot
            grid.append(
                {
                    "slot": absolute_slot,
                    "time_sec": round(0.1974 + absolute_slot * slot_sec, 6),
                    "bar": bar + 1,
                    "beat_in_bar": slot // 4 + 1,
                    "subdivision": slot % 4,
                }
            )
    return {
        "audio_file": "120_trap.mp3",
        "time_signature": "4/4",
        "bpm": {"fixed_integer": bpm},
        "absolute_grid": grid,
    }


class FlowPlannerTests(unittest.TestCase):
    def test_korean_phonemes(self):
        self.assertEqual(hangul_to_phonemes("각"), ["g", "a", "kcl"])

    def test_plan_uses_absolute_grid(self):
        lyrics = ["나는 비트 위를 달려"] * 8
        plan = build_flow_plan(
            make_analysis(),
            lyrics,
            "trap",
            "potg",
            normalizer=lambda text: text,
        )
        self.assertEqual(plan["bars"][0]["startSec"], 0.1974)
        self.assertEqual(plan["bars"][1]["startSec"], 2.1974)
        self.assertEqual(plan["bars"][0]["snareSlots"], [8])
        self.assertTrue(any(item["accent"] for item in plan["bars"][0]["placements"]))

    def test_diffsinger_duration_fills_each_bar(self):
        plan = build_flow_plan(
            make_analysis(),
            ["나는 비트 위를 달려"] * 8,
            "trap",
            "potg",
            normalizer=lambda text: text,
        )
        score = to_diffsinger_ds(plan, 190.0)
        first_duration = sum(float(value) for value in score[0]["ph_dur"].split())
        self.assertAlmostEqual(first_duration, 2.0, places=5)
        self.assertEqual(
            sum(int(value) for value in score[0]["ph_num"].split()),
            len(score[0]["ph_seq"].split()),
        )


if __name__ == "__main__":
    unittest.main()
