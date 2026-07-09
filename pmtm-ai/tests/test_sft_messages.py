import io
import sys
import unittest
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.prepare_dataset import prepare
from app.training.sft_qwen import _tokenize_messages


class FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = []
        for message in messages:
            rendered.append(f"<{message['role']}>\n{message['content']}\n</{message['role']}>\n")
        if add_generation_prompt:
            rendered.append("<assistant>\n")
        return "".join(rendered)

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        ids = list(range(len(text)))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }


class SftMessagesTests(unittest.TestCase):
    def test_prepare_outputs_chat_messages_with_assistant_lyrics(self):
        df = pd.DataFrame([
            {
                "artist": "artist",
                "lyrics": "\n".join(f"line {i}" for i in range(1, 9)),
                "bpm": 90,
                "energy": 0.5,
                "danceability": 0.5,
                "loudness": -6.0,
                "valence": 0.5,
            }
        ])

        records = prepare(df)

        self.assertEqual(len(records), 1)
        messages = records[0]["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertIn("BPM 90", messages[0]["content"])
        self.assertIn("exactly 8 lines", messages[0]["content"])
        self.assertTrue(messages[1]["content"].endswith("[End]"))

    def test_tokenize_messages_masks_user_tokens(self):
        messages = [
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "assistant"},
        ]
        tokenizer = FakeTokenizer()

        encoded = _tokenize_messages(tokenizer, messages, max_length=1000)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        prompt_len = len(prompt_text)

        self.assertTrue(all(label == -100 for label in encoded["labels"][:prompt_len]))
        self.assertTrue(any(label != -100 for label in encoded["labels"][prompt_len:]))
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))


if __name__ == "__main__":
    unittest.main()
