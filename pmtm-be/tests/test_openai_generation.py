import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main


class _FakeArray:
    def __init__(self, value):
        self.value = value

    def reshape(self, *_args):
        return [self.value]


class _FakeNumpy:
    @staticmethod
    def asarray(value):
        return _FakeArray(value)


class BeatGenerationTests(unittest.TestCase):
    def test_analyze_beat_bpm_rounds_librosa_tempo(self):
        with NamedTemporaryFile(suffix=".wav") as audio:
            with mock.patch.dict(
                "sys.modules",
                {
                    "librosa": mock.Mock(
                        load=mock.Mock(return_value=([1.0] * 1024, 22050)),
                        beat=mock.Mock(beat_track=mock.Mock(return_value=(89.6, []))),
                    ),
                    "numpy": _FakeNumpy,
                },
            ):
                self.assertEqual(main._analyze_beat_bpm(audio.name), 90)

    def test_analyze_beat_bpm_rejects_out_of_range_tempo(self):
        with NamedTemporaryFile(suffix=".wav") as audio:
            with mock.patch.dict(
                "sys.modules",
                {
                    "librosa": mock.Mock(
                        load=mock.Mock(return_value=([1.0] * 1024, 22050)),
                        beat=mock.Mock(beat_track=mock.Mock(return_value=(300.0, []))),
                    ),
                    "numpy": _FakeNumpy,
                },
            ):
                with self.assertRaises(HTTPException) as exc:
                    main._analyze_beat_bpm(audio.name)

        self.assertEqual(exc.exception.status_code, 400)
        self.assertEqual(exc.exception.detail, "BPM 분석 실패")

    def test_generate_from_beat_uses_analyzed_bpm(self):
        client = TestClient(main.app)

        with (
            mock.patch.object(main, "_analyze_beat_bpm", return_value=92) as analyze,
            mock.patch.object(
                main,
                "_generate_verse_for_model",
                return_value=("[Verse]\n가사", ["model note"]),
            ) as generate,
        ):
            response = client.post(
                "/api/v1/lyrics/generate-from-beat",
                data={"llm": "qwen-local"},
                files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["bpm"], 92)
        self.assertIn("rhymeAnalysis", response.json())
        self.assertIn("librosa tempo 분석값을 BPM으로 사용했습니다.", response.json()["notes"])
        analyze.assert_called_once()
        generate.assert_called_once_with(92, "qwen-local")
        self.assertFalse(Path(analyze.call_args.args[0]).exists())

    def test_generate_from_beat_rejects_non_audio_file(self):
        client = TestClient(main.app)

        response = client.post(
            "/api/v1/lyrics/generate-from-beat",
            data={"llm": "qwen-local"},
            files={"beat": ("beat.txt", b"not audio", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "지원하지 않는 오디오 형식입니다.")

    def test_generate_from_beat_rejects_empty_file(self):
        client = TestClient(main.app)

        response = client.post(
            "/api/v1/lyrics/generate-from-beat",
            data={"llm": "qwen-local"},
            files={"beat": ("beat.wav", b"", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "비트 파일이 비어 있습니다.")


class RhymeAnalysisTests(unittest.TestCase):
    def test_analyze_rhyme_handles_empty_lines(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": []})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_analyze_rhyme_groups_similar_lines(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": ["강", "방"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["rhymeGroup"], body[1]["rhymeGroup"])
        self.assertIsNotNone(body[0]["rhymeGroup"])
        self.assertGreaterEqual(body[0]["score"], 0.72)
        self.assertEqual(body[0]["highlightStart"], 0)
        self.assertEqual(body[0]["highlightEnd"], 1)

    def test_analyze_rhyme_leaves_different_lines_ungrouped(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": ["강", "비트"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body[0]["rhymeGroup"])
        self.assertIsNone(body[1]["rhymeGroup"])

    def test_generate_lyrics_response_includes_rhyme_analysis(self):
        client = TestClient(main.app)

        with mock.patch.object(
            main,
            "_generate_verse_for_model",
            return_value=("[Verse]\n강\n방", ["model note"]),
        ):
            response = client.post("/api/v1/lyrics/generate", json={"bpm": 90, "llm": "qwen-local"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body["rhymeAnalysis"]), 2)
        self.assertEqual(body["rhymeAnalysis"][0]["text"], "강")


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
