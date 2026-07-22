import sys
import unittest
from pathlib import Path
from unittest import mock

# pyrefly: ignore [missing-import]
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lyric_prompts import build_messages
from app.training.grpo_qwen import _format_finite_summary
from app.training.grpo_qwen import _summarize_tensors, build_prompts, rhyme_reward


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

        self.assertEqual(len(prompts), 2)
        
        # 1st prompt (bpm=90, 붐뱁)
        msg1 = prompts[0]
        self.assertEqual([m["role"] for m in msg1], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", msg1[0]["content"])
        self.assertIn("6~18 범위 내로", msg1[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg1[0]["content"])

        # 2nd prompt (bpm=140, 트랩)
        msg2 = prompts[1]
        self.assertEqual([m["role"] for m in msg2], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", msg2[0]["content"])
        self.assertIn("6~18 범위 내로", msg2[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg2[0]["content"])

    def test_build_prompts_doubles_halftime_bpm(self):
        # 70 BPM should be doubled to 140 BPM, resulting in "트랩" (6~18 syllables)
        messages = build_messages(bpm=70)
        self.assertEqual(len(messages), 1)
        self.assertEqual([m["role"] for m in messages], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("6~18 범위 내로", messages[0]["content"])

    def test_rhyme_reward_accepts_conversational_completion(self):
        prompt = build_messages(bpm=90)
        raw_lines = [
            "밤을 지나 나는 다시 올라가",
            "맘을 비워도 박자는 돌아가",
            "길 위의 불빛이 나를 불러가",
            "진심을 눌러도 rhyme은 흘러가",
            "차가운 빗줄기가 어깨에 내렸지",
            "이 밤이 흐르기 전 모든 걸 바쳤지",
            "과거의 기억들을 저 멀리 버렸지",
            "내 앞을 막아선 저 쇠창살을 막았지",
        ]
        formatted_lines = [f"{i}. ({len(ln)}음절) {ln}" for i, ln in enumerate(raw_lines, 1)]
        content = "\n".join(formatted_lines)
        
        completion = [[
            {
                "role": "assistant",
                "content": content,
            }
        ]]

        rewards = rhyme_reward(completion, prompts=[prompt])

        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)
        # 이 가사는 AAAA BBBB 형태이므로 높은 점수(>0.75)가 기대됨
        self.assertGreater(rewards[0], 0.70)

    def test_rhyme_reward_penalizes_abab_and_rewards_aa_and_aaaa(self):
        # 1. AAAA BBBB 가사 (1~4행 -가 라임, 5~8행 -다 라임)
        lines_aaaa_bbbb = [
            "밤을 지나 나는 다시 올라가",
            "맘을 비워도 박자는 돌아가",
            "길 위의 불빛이 나를 불러가",
            "진심을 눌러도 rhyme은 흘러가",
            "차가운 빗줄기가 어깨에 내렸다",
            "이 밤이 흐르기 전 모든 걸 바쳤다",
            "과거의 기억들을 저 멀리 버렸다",
            "내 앞을 막아선 저 쇠창살을 막았다",
        ]
        
        # 2. AA BB CC DD 가사 (2줄 단위 라임)
        lines_aabb_ccdd = [
            "밤을 지나 나는 다시 올라가",
            "맘을 비워도 박자는 돌아가",
            "차가운 빗줄기가 어깨에 내렸지",
            "이 밤이 흐르기 전 모든 걸 바쳤지",
            "그 누구도 내 앞길을 막지 못해",
            "끝까지 가겠어 난 절대 안 멈춰",  # 라임 안맞음 (못해/멈춰)
            "새로운 세상을 향해서 가겠어",
            "꿈을 향해 한 걸음 더 내딛겠어",  # 가겠어/내딛겠어 (라임 맞음 -어)
        ]

        # 3. ABAB CDCD 가사 (교차 라임)
        lines_abab_cdcd = [
            "밤을 지나 나는 다시 올라가",      # A
            "차가운 빗줄기가 어깨에 내렸다",    # B
            "맘을 비워도 박자는 돌아가",        # A
            "이 밤이 흐르기 전 모든 걸 바쳤다",  # B
            "그 누구도 내 앞길을 막지 못해",    # C
            "새로운 세상을 향해서 가겠어",      # D
            "이 모든 두려움을 뛰어 넘게",      # C
            "꿈을 향해 한 걸음 더 내딛겠어",    # D
        ]

        def format_comp(lines):
            formatted = [f"{i}. ({len(ln)}음절) {ln}" for i, ln in enumerate(lines, 1)]
            return [[{"role": "assistant", "content": "\n".join(formatted)}]]

        prompt = build_messages(bpm=90)
        
        reward_aaaa = rhyme_reward(format_comp(lines_aaaa_bbbb), prompts=[prompt])[0]
        reward_aabb = rhyme_reward(format_comp(lines_aabb_ccdd), prompts=[prompt])[0]
        reward_abab = rhyme_reward(format_comp(lines_abab_cdcd), prompts=[prompt])[0]
        # AAAA BBBB는 0.70 이상의 높은 점수여야 함
        self.assertGreater(reward_aaaa, 0.70)
        # AA BB CC DD는 중간 정도의 점수여야 함
        self.assertTrue(0.4 <= reward_aabb <= 0.85)
        # ABAB CDCD는 상대적으로 낮아야 함
        self.assertLess(reward_abab, 0.65)
        # 서열 관계 검증: 각각 AAAA보다는 작아야 함
        self.assertLess(reward_abab, reward_aaaa)
        self.assertLess(reward_aabb, reward_aaaa)


if __name__ == "__main__":
    unittest.main()
