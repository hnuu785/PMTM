import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import generate, generate_for_api


class FakeTokenizer:
    chat_template = "fake-template"
    name_or_path = "fake-tokenizer"

    def __init__(self):
        self.last_messages = None
        self.last_tokenize = None
        self.last_add_generation_prompt = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.last_messages = messages
        self.last_tokenize = tokenize
        self.last_add_generation_prompt = add_generation_prompt
        return "CHAT_PROMPT"


class InferencePromptingTests(unittest.TestCase):
    def test_auto_prompt_format_uses_chat_for_instruct_models_only(self):
        self.assertTrue(generate_for_api.should_use_chat_template("Qwen/Qwen2.5-3B-Instruct", "auto"))
        self.assertFalse(generate_for_api.should_use_chat_template("Qwen/Qwen2.5-3B", "auto"))
        self.assertTrue(generate_for_api.should_use_chat_template("Qwen/Qwen2.5-3B", "chat"))
        self.assertFalse(generate_for_api.should_use_chat_template("Qwen/Qwen2.5-3B-Instruct", "raw"))

    def test_chat_prompt_uses_tokenizer_chat_template(self):
        tokenizer = FakeTokenizer()
        prompt = generate_for_api.build_model_input_text(
            tokenizer,
            "Qwen/Qwen2.5-3B-Instruct",
            "auto",
            "RAW_PROMPT",
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(prompt, "CHAT_PROMPT")
        self.assertEqual(tokenizer.last_messages, [{"role": "user", "content": "hello"}])
        self.assertFalse(tokenizer.last_tokenize)
        self.assertTrue(tokenizer.last_add_generation_prompt)

    def test_raw_prompt_keeps_legacy_adapter_prompt(self):
        tokenizer = FakeTokenizer()
        prompt = generate_for_api.build_model_input_text(
            tokenizer,
            "Qwen/Qwen2.5-1.5B",
            "auto",
            "RAW_PROMPT",
            [{"role": "user", "content": "hello"}],
        )

        self.assertEqual(prompt, "RAW_PROMPT")
        self.assertIsNone(tokenizer.last_messages)

    def test_api_messages_request_eight_line_lyrics(self):
        args = SimpleNamespace(bpm=90, genre="Korean hip-hop", mood="confident", bars=8)
        messages = generate_for_api.build_messages(args)

        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("exactly 8 lines", messages[0]["content"])
        self.assertNotIn("genre", messages[0]["content"])
        self.assertNotIn("mood", messages[0]["content"])

    def test_local_cli_messages_use_training_prompt_shape(self):
        args = SimpleNamespace(
            artist="Tablo",
            bpm=90,
            energy=0.65,
            danceability=0.70,
            loudness=-6.0,
            valence=0.50,
            bars=8,
        )
        messages = generate.build_messages(args)

        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("BPM 90", messages[0]["content"])
        self.assertIn("exactly 8 lines", messages[0]["content"])
        self.assertNotIn("Tablo", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
