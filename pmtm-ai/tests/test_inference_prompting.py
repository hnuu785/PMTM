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
from app.inference.device import select_inference_device


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


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeMps:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    float16 = "float16"
    float32 = "float32"

    def __init__(self, *, cuda_available=False, mps_available=False):
        self.cuda = FakeCuda(cuda_available)
        self.backends = SimpleNamespace(mps=FakeMps(mps_available))


class InferencePromptingTests(unittest.TestCase):
    def test_inference_device_prefers_cuda(self):
        self.assertEqual(
            select_inference_device(FakeTorch(cuda_available=True, mps_available=True)),
            ("cuda", "float16"),
        )

    def test_inference_device_uses_mps_when_cuda_is_unavailable(self):
        self.assertEqual(
            select_inference_device(FakeTorch(mps_available=True)),
            ("mps", "float16"),
        )

    def test_inference_device_falls_back_to_cpu(self):
        self.assertEqual(select_inference_device(FakeTorch()), ("cpu", "float32"))

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
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("8~16 범위 내로", messages[0]["content"])

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
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("8~16 범위 내로", messages[0]["content"])

    def test_local_cli_messages_omit_bpm_when_unspecified(self):
        args = SimpleNamespace(bpm=None, bars=None)
        messages = generate.build_messages(args)

        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertIn("랩 가사를 작성해 주세요", messages[0]["content"])
        self.assertIn("정확히 16줄로 구성해야 하며", messages[0]["content"])
        self.assertIn("6~14 범위 내로", messages[0]["content"])

    def test_bpm_threshold_distinction(self):
        # 114 BPM should be Boombap (8 lines, 8~16 syllables)
        args_114 = SimpleNamespace(bpm=114, bars=None)
        messages_114 = generate.build_messages(args_114)
        self.assertIn("정확히 8줄로 구성해야 하며", messages_114[0]["content"])
        self.assertIn("8~16 범위 내로", messages_114[0]["content"])

        # 115 BPM should be Trap (16 lines, 6~14 syllables)
        args_115 = SimpleNamespace(bpm=115, bars=None)
        messages_115 = generate.build_messages(args_115)
        self.assertIn("정확히 16줄로 구성해야 하며", messages_115[0]["content"])
        self.assertIn("6~14 범위 내로", messages_115[0]["content"])

    def test_parse_target_bars_korean_prompt(self):
        import re
        ko_lines_re = re.compile(r"정확히\s+(\d+)줄")
        prompt_16 = "트랩 랩 가사를 작성해 주세요. 정확히 16줄로 구성해야 하며, 줄당 음절 수는 6~14 범위 내로 조절해 주세요."
        prompt_8 = "붐뱁 랩 가사를 작성해 주세요. 정확히 8줄로 구성해야 하며, 줄당 음절 수는 8~16 범위 내로 조절해 주세요."
        self.assertEqual(int(ko_lines_re.search(prompt_16).group(1)), 16)
        self.assertEqual(int(ko_lines_re.search(prompt_8).group(1)), 8)


if __name__ == "__main__":
    unittest.main()

