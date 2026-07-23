import io
import sys
import unittest
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.training.prepare_dataset_v3 import build_record
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
        lines = [f"line {i}" for i in range(1, 9)]
        record = build_record(lines, "붐뱁", 90.0)

        messages = record["messages"]
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("8~16 범위 내로", messages[0]["content"])
        self.assertIn("1. (2음절) line 1", messages[1]["content"])
        self.assertIn("8. (2음절) line 8", messages[1]["content"])


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
