import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flow_planner_merged import (
    align_original_pronunciation,
    build_flow_plan,
    calculate_micro_offset,
    to_diffsinger_ds,
    validate_outputs,
)


class FakeToken:
    def __init__(self, form, tag, start, length):
        self.form = form
        self.tag = tag
        self.start = start
        self.len = length


class FakeKiwi:
    def tokenize(self, text):
        return [
            FakeToken(char, "NNG" if index == 0 else "JX", index, 1)
            for index, char in enumerate(text)
            if "가" <= char <= "힣"
        ]


def make_analysis(bpm=120, bars=8, with_snare=False):
    slot_sec = 60 / bpm / 4
    grid = []
    for bar in range(bars):
        for slot in range(16):
            absolute_slot = bar * 16 + slot
            grid.append(
                {
                    "slot": absolute_slot,
                    "time_sec": round(0.2 + absolute_slot * slot_sec, 6),
                    "bar": bar + 1,
                }
            )
    analysis = {
        "audio_file": "test.wav",
        "time_signature": "4/4",
        "bpm": {"fixed_integer": bpm},
        "absolute_grid": grid,
    }
    if with_snare:
        analysis["snare_detection"] = {
            "events": [
                {
                    "grid_slot": 7,
                    "confidence": 0.9,
                    "original_time": 1.21,
                    "snapped_time": 1.2,
                }
            ]
        }
    return analysis


class MergedFlowPlannerTests(unittest.TestCase):
    def test_alignment_preserves_original_and_pronounced_syllables(self):
        analysis = {
            "morphemes": [
                {
                    "id": 0,
                    "form": "접어",
                    "tag": "VV",
                    "start": 0,
                    "end": 2,
                    "isContent": True,
                    "isFunction": False,
                }
            ],
            "eojeols": [{"id": 0, "surface": "접어", "start": 0, "end": 2}],
        }
        alignment, units = align_original_pronunciation("접어", "저버", analysis)
        self.assertEqual([unit["pronounced"] for unit in units], ["저", "버"])
        self.assertEqual([unit["original"] for unit in units], ["접", "어"])
        self.assertEqual([item["operation"] for item in alignment], ["substitute", "substitute"])

    def test_micro_offset_preserves_flow_b_rule(self):
        unit = {"absoluteSlot": 7, "stress": 1.2}
        offset, reason, snare = calculate_micro_offset(
            unit,
            "trap",
            0.125,
            {
                7: {
                    "confidence": 0.9,
                    "originalTimeSec": 1.21,
                    "snappedTimeSec": 1.2,
                }
            },
        )
        self.assertEqual(offset, -7.5)
        self.assertEqual(reason, "snare")
        self.assertIsNotNone(snare)

    def test_merged_plan_exports_final_timing(self):
        lyrics = ["나는 비트 위를 달려"] * 8
        plan = build_flow_plan(
            make_analysis(with_snare=True),
            lyrics,
            "trap",
            "potg",
            normalizer=lambda text: text,
            kiwi=FakeKiwi(),
        )
        score = to_diffsinger_ds(plan)
        validate_outputs(plan, score)
        first_bar = plan["bars"][0]
        snare_units = [
            item for item in first_bar["placements"] if item["absoluteSlot"] == 7
        ]
        self.assertEqual(len(snare_units), 1)
        self.assertEqual(snare_units[0]["microOffsetReason"], "snare")
        self.assertNotEqual(
            snare_units[0]["gridStartSec"], snare_units[0]["finalStartSec"]
        )
        self.assertEqual(len(score), 8)
        self.assertIn("D4", score[0]["note_seq"])
        self.assertAlmostEqual(
            sum(float(value) for value in score[0]["ph_dur"].split()),
            2.0,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
