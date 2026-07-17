import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lyric_prompts import build_api_messages
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
        self.assertIn("랩을 작성해 주세요", msg1[0]["content"])
        self.assertIn("10~14 범위 내로", msg1[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg1[0]["content"])

        # 2nd prompt (bpm=140, 트랩)
        msg2 = prompts[1]
        self.assertEqual([m["role"] for m in msg2], ["user"])
        self.assertIn("랩을 작성해 주세요", msg2[0]["content"])
        self.assertIn("14~18 범위 내로", msg2[0]["content"])
        self.assertNotIn("AAAABBBB 스키마를 준수", msg2[0]["content"])

    def test_build_prompts_doubles_halftime_bpm(self):
        # 70 BPM should be doubled to 140 BPM, resulting in "트랩" (14~18 syllables)
        messages = build_api_messages(bpm=70)
        self.assertEqual(len(messages), 1)
        self.assertIn("랩을 작성해 주세요", messages[0]["content"])
        self.assertIn("14~18 범위 내로", messages[0]["content"])

    def test_rhyme_reward_accepts_conversational_completion(self):
        prompt = build_api_messages(bpm=90)
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
        formatted_lines = [f"{i}. {ln} ({len(ln)}음절)" for i, ln in enumerate(raw_lines, 1)]
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




if __name__ == "__main__":
    unittest.main()
