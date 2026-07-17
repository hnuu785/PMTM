import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lyric_prompts import build_api_messages
from app.training.grpo_qwen import _format_finite_summary, _repeated_ngram_ratio, _short_line_ratio
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
        df = pd.DataFrame([
            {"bpm": 90, "rhyme_density": 0.7, "title": "test_title"},
        ])

        prompts = build_prompts(df)

        self.assertEqual(len(prompts), 1)
        messages = prompts[0]
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("붐뱁 장르의 한국어 랩 가사", messages[0]["content"])
        self.assertIn("AAAABBBB 스키마를 준수", messages[0]["content"])

    def test_rhyme_reward_accepts_conversational_completion(self):
        prompt = build_api_messages(bpm=90, rhyme_scheme="AAAABBBB")
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

    def test_rhyme_reward_penalizes_repeated_ngrams(self):
        prompt = build_api_messages(bpm=90, rhyme_scheme="AAAABBBB")
        repeated_lines = [
            "나는 계속 달려가 밤길",
            "나는 계속 달려가 별빛",
            "나는 계속 달려가 거리",
            "나는 계속 달려가 소리",
            "나는 계속 달려가 무대",
            "나는 계속 달려가 열기",
            "나는 계속 달려가 숨결",
            "나는 계속 달려가 내일",
        ]
        varied_lines = [
            "나는 조용히 걸어가 밤길",
            "우린 새벽을 지나 별빛",
            "발끝 위로 번져 거리",
            "숨을 고른 뒤 소리",
            "무대 앞에서 열기",
            "박자 사이로 숨결",
            "흔들림 없이 내일",
            "마지막 줄에 불빛",
        ]
        
        def format_content(lines):
            formatted = [f"{i}. {ln} ({len(ln)}음절)" for i, ln in enumerate(lines, 1)]
            return "\n".join(formatted)
            
        completions = [
            [{"role": "assistant", "content": format_content(repeated_lines)}],
            [{"role": "assistant", "content": format_content(varied_lines)}],
        ]

        self.assertGreater(_repeated_ngram_ratio(repeated_lines), _repeated_ngram_ratio(varied_lines))
        with mock.patch("app.training.grpo_qwen.get_line_rhyme_score", return_value=0.5):
            repeated_reward, varied_reward = rhyme_reward(completions, prompts=[prompt, prompt])

        self.assertLess(repeated_reward, varied_reward)


if __name__ == "__main__":
    unittest.main()
