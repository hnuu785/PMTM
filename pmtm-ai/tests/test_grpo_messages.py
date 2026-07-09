import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.lyric_prompts import build_api_messages
from app.training.grpo_qwen import _repeated_ngram_ratio, _short_line_ratio, build_prompts, rhyme_reward


class GrpoMessagesTests(unittest.TestCase):
    def test_build_prompts_outputs_chat_messages(self):
        df = pd.DataFrame([
            {"bpm": 90, "rhyme_density": 0.7},
        ])

        prompts = build_prompts(df)

        self.assertEqual(len(prompts), 1)
        messages = prompts[0]
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("BPM 90", messages[0]["content"])
        self.assertIn("exactly 8 lines", messages[0]["content"])

    def test_rhyme_reward_accepts_conversational_completion(self):
        prompt = build_api_messages(bpm=90)
        completion = [[
            {
                "role": "assistant",
                "content": "\n".join([
                    "밤을 지나 나는 다시 올라가",
                    "맘을 비워도 박자는 돌아가",
                    "길 위의 불빛이 나를 불러가",
                    "진심을 눌러도 rhyme은 흘러가",
                    "숨을 고르고 다음 줄로 걸어가",
                    "흔들려도 내 발은 앞으로 가",
                    "끝을 모르지만 계속 써 내려가",
                    "무대 위에서 내 이름을 불러봐",
                    "[End]",
                ]),
            }
        ]]

        rewards = rhyme_reward(completion, prompts=[prompt])

        self.assertEqual(len(rewards), 1)
        self.assertIsInstance(rewards[0], float)

    def test_rhyme_reward_penalizes_repeated_ngrams(self):
        prompt = build_api_messages(bpm=90)
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
        completions = [
            [{"role": "assistant", "content": "\n".join([*repeated_lines, "[End]"])}],
            [{"role": "assistant", "content": "\n".join([*varied_lines, "[End]"])}],
        ]

        self.assertGreater(_repeated_ngram_ratio(repeated_lines), _repeated_ngram_ratio(varied_lines))
        with mock.patch("app.training.grpo_qwen.get_line_rhyme_score", return_value=0.5):
            repeated_reward, varied_reward = rhyme_reward(completions, prompts=[prompt, prompt])

        self.assertLess(repeated_reward, varied_reward)

    def test_rhyme_reward_penalizes_short_lines(self):
        prompt = build_api_messages(bpm=90)
        short_lines = [
            "야",
            "빛을 따라 걸어가",
            "왜",
            "박자 위로 올라가",
            "숨을 고르고 말해",
            "무대 앞에 서 있어",
            "끝을 보고 달려가",
            "내일 쪽으로 걸어가",
        ]
        full_lines = [
            "밤을 지나 나는 다시 올라가",
            "맘을 비워도 박자는 돌아가",
            "길 위의 불빛이 나를 불러가",
            "진심을 눌러도 rhyme은 흘러가",
            "숨을 고르고 다음 줄로 걸어가",
            "흔들려도 내 발은 앞으로 가",
            "끝을 모르지만 계속 써 내려가",
            "무대 위에서 내 이름을 불러봐",
        ]
        completions = [
            [{"role": "assistant", "content": "\n".join([*short_lines, "[End]"])}],
            [{"role": "assistant", "content": "\n".join([*full_lines, "[End]"])}],
        ]

        self.assertGreater(_short_line_ratio(short_lines, 8), _short_line_ratio(full_lines, 8))
        with mock.patch("app.training.grpo_qwen.get_line_rhyme_score", return_value=0.5):
            short_reward, full_reward = rhyme_reward(completions, prompts=[prompt, prompt])

        self.assertLess(short_reward, full_reward)


if __name__ == "__main__":
    unittest.main()
