import sys
import unittest
from pathlib import Path
from unittest import mock

# pyrefly: ignore [missing-import]
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lyric_prompts import build_messages
from app.rhyme_scoring.rhyme_engine import analyze_bar_end_rhyme
from app.training.grpo_qwen import _format_finite_summary
from app.training.grpo_qwen import _summarize_tensors, build_prompts, format_reward, rhyme_reward
from app.rhyme_scoring.phonetics_utils import count_syllables


class GrpoMessagesTests(unittest.TestCase):
    def test_summarize_tensors_detects_nonfinite_values(self):
        summary = _summarize_tensors([
            ("ok", torch.tensor([1.0, -2.0])),
            ("bad", torch.tensor([float("nan"), float("inf")])),
        ])

        self.assertFalse(summary.ok)
        self.assertEqual(summary.total, 4)
        self.assertEqual(summary.nonfinite, 2)
        self.assertEqual(summary.max_abs, 2.0)
        self.assertEqual(summary.examples, ["bad"])
        self.assertIn("BAD", _format_finite_summary("test", summary))

    def test_build_prompts_outputs_chat_messages(self):
        prompts = build_prompts()

        self.assertEqual(len(prompts), 8)
        
        # 1st prompt (bpm=90, 붐뱁)
        msg1 = prompts[0]
        self.assertEqual([m["role"] for m in msg1], ["user"])
        self.assertIn("한국어 중심의", msg1[0]["content"])
        self.assertIn("랩 가사를 작성해 주세요", msg1[0]["content"])
        self.assertIn("9~16 범위 내로", msg1[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg1[0]["content"])

        # 140 BPM, 트랩 prompt (5th prompt in 2x4 combination)
        msg2 = prompts[4]
        self.assertEqual([m["role"] for m in msg2], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", msg2[0]["content"])
        self.assertIn("6~13 범위 내로", msg2[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg2[0]["content"])

    def test_build_prompts_doubles_halftime_bpm(self):
        # 70 BPM should be doubled to 140 BPM, resulting in "트랩" (6~13 syllables)
        messages = build_messages(bpm=70)
        self.assertEqual(len(messages), 1)
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("6~13 범위 내로", messages[0]["content"])

    @staticmethod
    def _line(index: int, ending: str) -> str:
        prefixes = "가나다라마바사아자차카타파하거너더러"
        return f"{prefixes[index]}라마바사아자차카 {ending}"

    @staticmethod
    def _completion(lines: list[str]):
        content = "\n".join(
            f"{index}. ({count_syllables(line)}음절) {line}"
            for index, line in enumerate(lines, 1)
        )
        return [[{"role": "assistant", "content": content}]]

    @staticmethod
    def _same_ending_rhyme(line1: str, line2: str) -> float:
        return 1.0 if line1.rsplit(maxsplit=1)[-1] == line2.rsplit(maxsplit=1)[-1] else 0.0

    def test_format_reward_requires_exact_contract(self):
        prompt = build_messages(bpm=90)
        lines = [self._line(index, "가") for index in range(8)]
        valid = self._completion(lines)
        self.assertEqual(format_reward(valid, prompts=[prompt]), [1.0])

        content = valid[0][0]["content"]
        trailing_tag = "\n".join(
            f"{index}. {line.rsplit(' ', 1)[-1]} ({count_syllables(lines[index - 1])}음절)"
            for index, line in enumerate(lines, 1)
        )
        header = f"가사입니다:\n{content}"

        self.assertLess(format_reward([[{"role": "assistant", "content": trailing_tag}]], prompts=[prompt])[0], 1.0)
        self.assertLess(format_reward([[{"role": "assistant", "content": header}]], prompts=[prompt])[0], 1.0)
        self.assertEqual(rhyme_reward([[{"role": "assistant", "content": header}]], prompts=[prompt]), [0.0])

    def test_rhyme_reward_gives_aabb_and_aaaa_the_same_full_score(self):
        prompt = build_messages(bpm=90)
        aabb_lines = [self._line(index, ending) for index, ending in enumerate("가가나나다다라라")]
        aaaa_lines = [self._line(index, ending) for index, ending in enumerate("가가가가나나나나")]

        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            aabb_reward = rhyme_reward(self._completion(aabb_lines), prompts=[prompt])[0]
            aaaa_reward = rhyme_reward(self._completion(aaaa_lines), prompts=[prompt])[0]

        self.assertEqual(aabb_reward, 1.0)
        self.assertEqual(aaaa_reward, 1.0)

    def test_rhyme_reward_penalizes_uncovered_endings(self):
        prompt = build_messages(bpm=90)
        aaabccdd_lines = [self._line(index, ending) for index, ending in enumerate("가가가나다다라라")]
        abab_lines = [self._line(index, ending) for index, ending in enumerate("가나가나다라다라")]

        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            aaabccdd_reward = rhyme_reward(self._completion(aaabccdd_lines), prompts=[prompt])[0]
            abab_reward = rhyme_reward(self._completion(abab_lines), prompts=[prompt])[0]

        self.assertEqual(aaabccdd_reward, 0.875)
        self.assertEqual(abab_reward, 0.0)

    def test_trap_rhyme_uses_even_numbered_lines_only(self):
        prompt = build_messages(bpm=140)
        even_endings = "가가나나다다라라"
        lines = []
        for index in range(16):
            ending = "고" if index % 2 == 0 else even_endings[index // 2]
            lines.append(self._line(index, ending))

        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            reward = rhyme_reward(self._completion(lines), prompts=[prompt])[0]
            analysis = analyze_bar_end_rhyme(lines, bpm=140)

        self.assertEqual(reward, 1.0)
        self.assertEqual(analysis.selected_line_indexes, (1, 3, 5, 7, 9, 11, 13, 15))
        self.assertTrue(all(analysis.rhyme_groups[index] is None for index in range(0, 16, 2)))
        self.assertTrue(all(analysis.rhyme_groups[index] is not None for index in range(1, 16, 2)))

    def test_shared_analysis_uses_all_boombap_lines(self):
        lines = [self._line(index, ending) for index, ending in enumerate("가가나나다다라라")]

        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            analysis = analyze_bar_end_rhyme(lines, bpm=90)

        self.assertEqual(analysis.selected_line_indexes, tuple(range(8)))
        self.assertTrue(all(group is not None for group in analysis.rhyme_groups))

    def test_rhyme_reward_penalizes_duplicate_lines(self):
        prompt = build_messages(bpm=90)
        lines = [self._line(index, "가") for index in range(8)]
        duplicated = list(lines)
        duplicated[-1] = duplicated[0]

        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            unique_reward = rhyme_reward(self._completion(lines), prompts=[prompt])[0]
            duplicate_reward = rhyme_reward(self._completion(duplicated), prompts=[prompt])[0]

        self.assertEqual(unique_reward, 1.0)
        self.assertEqual(duplicate_reward, 0.8)



    def test_shared_analysis_groups_non_adjacent_gap_1_rhymes(self):
        # ABAC
        lines_abac = [self._line(index, ending) for index, ending in enumerate("가나가다")]
        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            analysis_abac = analyze_bar_end_rhyme(lines_abac, bpm=90, max_gap=2)
        self.assertEqual(analysis_abac.rhyme_groups[0], analysis_abac.rhyme_groups[2])
        self.assertIsNotNone(analysis_abac.rhyme_groups[0])
        self.assertIsNone(analysis_abac.rhyme_groups[1])
        self.assertIsNone(analysis_abac.rhyme_groups[3])

        # AABAC
        lines_aabac = [self._line(index, ending) for index, ending in enumerate("가가나가다")]
        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            analysis_aabac = analyze_bar_end_rhyme(lines_aabac, bpm=90, max_gap=2)
        self.assertEqual(analysis_aabac.rhyme_groups[0], analysis_aabac.rhyme_groups[1])
        self.assertEqual(analysis_aabac.rhyme_groups[1], analysis_aabac.rhyme_groups[3])
        self.assertIsNotNone(analysis_aabac.rhyme_groups[0])

        # AABBCB
        lines_aabbcb = [self._line(index, ending) for index, ending in enumerate("가가나나다나")]
        with mock.patch(
            "app.rhyme_scoring.rhyme_engine.get_line_rhyme_score",
            side_effect=self._same_ending_rhyme,
        ):
            analysis_aabbcb = analyze_bar_end_rhyme(lines_aabbcb, bpm=90, max_gap=2)
        self.assertEqual(analysis_aabbcb.rhyme_groups[0], analysis_aabbcb.rhyme_groups[1])
        self.assertEqual(analysis_aabbcb.rhyme_groups[2], analysis_aabbcb.rhyme_groups[3])
        self.assertEqual(analysis_aabbcb.rhyme_groups[3], analysis_aabbcb.rhyme_groups[5])
        self.assertIsNotNone(analysis_aabbcb.rhyme_groups[2])
        self.assertNotEqual(analysis_aabbcb.rhyme_groups[0], analysis_aabbcb.rhyme_groups[2])


if __name__ == "__main__":
    unittest.main()

