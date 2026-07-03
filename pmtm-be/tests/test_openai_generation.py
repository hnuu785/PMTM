import unittest

from app import main


class OpenAIGenerationTests(unittest.TestCase):
    def test_openai_payload_limits_gpt5_reasoning(self):
        original_model = main.settings.openai_model
        main.settings.openai_model = "gpt-5-mini"

        try:
            payload = main._build_openai_payload(90)
        finally:
            main.settings.openai_model = original_model

        self.assertEqual(payload["max_output_tokens"], 500)
        self.assertEqual(payload["reasoning"], {"effort": "minimal"})
        self.assertEqual(payload["text"], {"verbosity": "low"})

    def test_openai_payload_requests_korean_lyrics(self):
        payload = main._build_openai_payload(90)

        self.assertIn("Korean rap lyrics", payload["instructions"])
        self.assertIn("한국어 랩", payload["input"])
        self.assertIn("대부분은 한국어", payload["input"])

    def test_extract_openai_text_from_output_content(self):
        data = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "one\n"},
                        {"type": "output_text", "text": "two"},
                    ]
                }
            ]
        }

        self.assertEqual(main._extract_openai_text(data), "one\n\ntwo")

    def test_empty_openai_response_describes_incomplete_details(self):
        message = main._describe_empty_openai_response(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            }
        )

        self.assertIn("status=incomplete", message)
        self.assertIn("max_output_tokens", message)


if __name__ == "__main__":
    unittest.main()
