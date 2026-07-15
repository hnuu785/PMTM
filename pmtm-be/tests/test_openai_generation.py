import unittest
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import mock

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import demo_pipeline
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

    def test_analyze_beat_returns_full_analysis_and_removes_upload(self):
        client = TestClient(main.app)
        analysis = {
            "fileName": "beat.wav",
            "durationSec": 12.5,
            "sampleRate": 22050,
            "sampleCount": 275625,
            "tempo": 92.4,
            "timeSignature": "4/4",
            "timeSignatureSource": "assumed",
            "introStartSec": 0.0,
            "introEndSec": 4.1,
            "drumEntrySec": 4.05,
            "firstBeatSec": 4.1,
            "firstBarStartSec": 4.1,
            "firstBarEndSec": 6.7,
            "firstBarBeatTimes": [4.1, 4.75, 5.4, 6.05],
            "beatTimes": [0.5, 1.15],
            "onsetTimes": [0.2, 0.5],
            "waveform": [{"time": 0.0, "value": 0.1}],
            "rms": [{"time": 0.0, "value": 0.2}],
            "onsetStrength": [{"time": 0.0, "value": 0.3}],
            "spectral": {"rmsMean": 0.2},
            "chroma": [0.1] * 12,
            "mfcc": [0.2] * 13,
        }

        with mock.patch.object(main, "_analyze_beat", return_value=analysis) as analyze:
            response = client.post(
                "/api/v1/beats/analyze",
                files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tempo"], 92.4)
        self.assertEqual(response.json()["timeSignature"], "4/4")
        self.assertEqual(response.json()["firstBeatSec"], 4.1)
        self.assertEqual(response.json()["firstBarEndSec"], 6.7)
        self.assertEqual(response.json()["chroma"], [0.1] * 12)
        analyze.assert_called_once()
        self.assertEqual(analyze.call_args.args[1], "beat.wav")
        self.assertFalse(Path(analyze.call_args.args[0]).exists())

    def test_select_sustained_onset_skips_isolated_intro_hit(self):
        frame = main._select_sustained_onset(
            onset_frames=[5, 40, 52, 64, 90],
            strengths=[0.8, 1.0, 0.9, 0.7, 0.8],
            horizon_frames=25,
        )

        self.assertEqual(frame, 40)

    def test_build_first_bar_uses_four_detected_beats_and_next_boundary(self):
        beats, end = main._build_first_bar(
            beat_times=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            first_beat_sec=1.0,
            tempo=120.0,
        )

        self.assertEqual(beats, [1.0, 1.5, 2.0, 2.5])
        self.assertEqual(end, 3.0)


class DemoGenerationTests(unittest.TestCase):
    def test_generate_demo_from_beat_enqueues_job(self):
        client = TestClient(main.app)

        with (
            TemporaryDirectory() as storage_dir,
            mock.patch.object(main, "DEMO_STORAGE_ROOT", Path(storage_dir)),
            mock.patch.object(main, "_enqueue_demo_job") as enqueue,
        ):
            response = client.post(
                "/api/v1/demos/generate-from-beat",
                data={
                    "llm": "qwen-local",
                    "genre": "trap",
                    "mood": "dark",
                    "demoLengthSec": "30",
                    "voice": "verse",
                    "vocalStartBars": "2",
                },
                files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "queued")
        self.assertTrue(body["jobId"])
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[3], "qwen-local")
        self.assertEqual(enqueue.call_args.args[4], "trap")
        self.assertEqual(enqueue.call_args.args[5], "dark")
        self.assertEqual(enqueue.call_args.args[6], 30)
        self.assertEqual(enqueue.call_args.args[7], "verse")
        self.assertEqual(enqueue.call_args.args[8], 2)

    def test_generate_demo_rejects_bad_length(self):
        client = TestClient(main.app)

        response = client.post(
            "/api/v1/demos/generate-from-beat",
            data={"demoLengthSec": "45", "voice": "verse"},
            files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "데모 길이는 30초 또는 60초만 지원합니다.")

    def test_generate_demo_rejects_bad_vocal_start_bars(self):
        client = TestClient(main.app)

        response = client.post(
            "/api/v1/demos/generate-from-beat",
            data={"demoLengthSec": "30", "voice": "verse", "vocalStartBars": "3"},
            files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "랩 시작 대기는 0, 2, 4, 8마디만 지원합니다.")

    def test_demo_status_serializes_redis_payload(self):
        client = TestClient(main.app)
        redis_client = mock.Mock()
        redis_client.hgetall.return_value = {
            "jobId": "abc",
            "status": "succeeded",
            "progress": "1.0",
            "bpm": "92",
            "lyrics": "[Verse]\n가사",
            "notes": '["done"]',
            "audioUrl": "/api/v1/demos/abc/audio",
        }

        with mock.patch.object(main, "_get_redis_client", return_value=redis_client):
            response = client.get("/api/v1/demos/abc")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "succeeded")
        self.assertEqual(body["bpm"], 92)
        self.assertEqual(body["notes"], ["done"])
        self.assertEqual(body["audioUrl"], "/api/v1/demos/abc/audio")
        self.assertIn("workerAvailable", body)

    def test_demo_status_warns_when_queued_without_worker(self):
        client = TestClient(main.app)
        redis_client = mock.Mock()
        redis_client.hgetall.return_value = {
            "jobId": "abc",
            "status": "queued",
            "progress": "0.0",
            "notes": '["queued"]',
        }

        with (
            mock.patch.object(main, "_get_redis_client", return_value=redis_client),
            mock.patch.object(main, "_get_demo_worker_count", return_value=0),
        ):
            response = client.get("/api/v1/demos/abc")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["workerAvailable"])
        self.assertEqual(body["workerCount"], 0)
        self.assertIn("데모 생성 워커가 실행 중이 아닙니다.", body["notes"][-1])

    def test_normalize_lyric_bars_pads_to_requested_length(self):
        bars = demo_pipeline.normalize_lyric_bars("[Verse]\none\ntwo", 4)

        self.assertEqual([bar.text for bar in bars], ["one", "two", "", ""])

    def test_synthesize_vocal_track_uses_one_file_per_bar(self):
        class FakeProvider:
            def synthesize_line(self, _text, _voice, _bpm, _bar_duration_sec, output_path):
                _write_test_wav(output_path, duration_sec=0.05)

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "vocal.wav"
            bars = [demo_pipeline.LyricBar(1, "one"), demo_pipeline.LyricBar(2, "two")]

            demo_pipeline.synthesize_vocal_track(FakeProvider(), bars, "verse", 120, output_path)

            self.assertTrue(output_path.exists())
            with wave.open(str(output_path), "rb") as audio:
                self.assertGreater(audio.getnframes(), 0)

    def test_synthesize_vocal_track_can_wait_before_first_bar(self):
        class FakeProvider:
            def synthesize_line(self, _text, _voice, _bpm, _bar_duration_sec, output_path):
                _write_test_wav(output_path, duration_sec=0.05)

        with TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "vocal.wav"
            bars = [demo_pipeline.LyricBar(1, "one")]

            demo_pipeline.synthesize_vocal_track(
                FakeProvider(),
                bars,
                "verse",
                120,
                output_path,
                start_offset_sec=0.5,
            )

            with wave.open(str(output_path), "rb") as audio:
                self.assertGreaterEqual(audio.getnframes(), int(0.55 * audio.getframerate()))

    def test_align_wav_to_duration_does_not_cut_long_lines(self):
        with TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "line.wav"
            output_path = Path(tmp) / "aligned.wav"
            _write_test_wav(input_path, duration_sec=0.4)

            demo_pipeline.align_wav_to_duration(input_path, output_path, target_duration_sec=0.2)

            with wave.open(str(output_path), "rb") as audio:
                self.assertEqual(audio.getnframes(), int(0.4 * audio.getframerate()))

    def test_run_demo_generation_marks_failed_when_provider_fails(self):
        class FakeRedisClient:
            def __init__(self):
                self.payload = {}

            def hset(self, _key, mapping):
                self.payload.update(mapping)

            def expire(self, *_args):
                pass

        fake_redis_client = FakeRedisClient()
        fake_redis_module = mock.Mock(Redis=mock.Mock(from_url=mock.Mock(return_value=fake_redis_client)))

        class FailingProvider:
            def synthesize_line(self, *_args):
                raise RuntimeError("provider failed")

        with (
            TemporaryDirectory() as tmp,
            mock.patch.dict("sys.modules", {"redis": fake_redis_module}),
            mock.patch.object(main, "_analyze_beat_bpm", return_value=90),
            mock.patch.object(main, "_generate_verse_for_model", return_value=("[Verse]\none", ["note"])),
            mock.patch.object(demo_pipeline, "_trim_beat_segment"),
            mock.patch.object(demo_pipeline, "get_vocal_provider", return_value=FailingProvider()),
        ):
            beat_path = Path(tmp) / "beat.wav"
            _write_test_wav(beat_path)
            demo_pipeline.run_demo_generation(
                "job",
                str(beat_path),
                tmp,
                "qwen-local",
                "trap",
                "dark",
                30,
                "verse",
            )

        self.assertEqual(fake_redis_client.payload["status"], "failed")
        self.assertIn("provider failed", fake_redis_client.payload["error"])

    def test_run_demo_generation_uses_eight_bars_for_sixty_second_demo(self):
        class FakeRedisClient:
            def hset(self, *_args, **_kwargs):
                pass

            def expire(self, *_args):
                pass

        fake_redis_module = mock.Mock(Redis=mock.Mock(from_url=mock.Mock(return_value=FakeRedisClient())))

        with (
            TemporaryDirectory() as tmp,
            mock.patch.dict("sys.modules", {"redis": fake_redis_module}),
            mock.patch.object(main, "_analyze_beat_bpm", return_value=90),
            mock.patch.object(main, "_generate_verse_for_model", return_value=("[Verse]\none", ["note"])) as generate,
            mock.patch.object(demo_pipeline, "_trim_beat_segment"),
            mock.patch.object(demo_pipeline, "get_vocal_provider", return_value=mock.Mock()),
            mock.patch.object(demo_pipeline, "synthesize_vocal_track", side_effect=RuntimeError("stop")),
        ):
            beat_path = Path(tmp) / "beat.wav"
            _write_test_wav(beat_path)
            demo_pipeline.run_demo_generation(
                "job",
                str(beat_path),
                tmp,
                "qwen-local",
                "trap",
                "dark",
                60,
                "verse",
            )

        self.assertEqual(generate.call_args.kwargs["bars"], 8)

    def test_run_demo_generation_passes_capped_vocal_start_offset(self):
        class FakeRedisClient:
            def hset(self, *_args, **_kwargs):
                pass

            def expire(self, *_args):
                pass

        fake_redis_module = mock.Mock(Redis=mock.Mock(from_url=mock.Mock(return_value=FakeRedisClient())))

        with (
            TemporaryDirectory() as tmp,
            mock.patch.dict("sys.modules", {"redis": fake_redis_module}),
            mock.patch.object(main, "_analyze_beat_bpm", return_value=90),
            mock.patch.object(main, "_generate_verse_for_model", return_value=("[Verse]\none", ["note"])),
            mock.patch.object(demo_pipeline, "_trim_beat_segment"),
            mock.patch.object(demo_pipeline, "get_vocal_provider", return_value=mock.Mock()),
            mock.patch.object(demo_pipeline, "synthesize_vocal_track", side_effect=RuntimeError("stop")) as synthesize,
        ):
            beat_path = Path(tmp) / "beat.wav"
            _write_test_wav(beat_path)
            demo_pipeline.run_demo_generation(
                "job",
                str(beat_path),
                tmp,
                "qwen-local",
                "trap",
                "dark",
                30,
                "verse",
                4,
            )

        self.assertAlmostEqual(synthesize.call_args.kwargs["start_offset_sec"], 30 - (8 * (60.0 / 90 * 4.0)))


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
        self.assertGreaterEqual(body[0]["score"], main.RHYME_GROUP_THRESHOLD)
        self.assertEqual(body[0]["highlightStart"], 0)
        self.assertEqual(body[0]["highlightEnd"], 1)
        self.assertEqual(body[0]["highlightRanges"], [{"start": 0, "end": 1}])

    def test_analyze_rhyme_highlights_matching_syllable_ranges(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": ["강물", "방물"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["highlightRanges"], [{"start": 0, "end": 2}])
        self.assertEqual(body[1]["highlightRanges"], [{"start": 0, "end": 2}])

    def test_analyze_rhyme_highlights_only_similar_syllables(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": ["달려가", "날아가"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["highlightRanges"], [{"start": 0, "end": 1}, {"start": 2, "end": 3}])
        self.assertEqual(body[1]["highlightRanges"], [{"start": 0, "end": 1}, {"start": 2, "end": 3}])

    def test_analyze_rhyme_leaves_different_lines_ungrouped(self):
        client = TestClient(main.app)

        response = client.post("/api/v1/lyrics/analyze-rhyme", json={"lines": ["강", "비트"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsNone(body[0]["rhymeGroup"])
        self.assertIsNone(body[1]["rhymeGroup"])
        self.assertEqual(body[0]["highlightRanges"], [])
        self.assertEqual(body[1]["highlightRanges"], [])

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
    def test_exp_005_sft_model_uses_adapter(self):
        with mock.patch.object(main, "_generate_qwen_verse", return_value="[Verse]\none") as generate:
            lyrics, notes = main._generate_verse_for_model(90, "qwen-exp-005-sft")

        self.assertEqual(lyrics, "[Verse]\none")
        generate.assert_called_once_with(
            90,
            main.EXP_005_SFT_ADAPTER,
            genre="Korean hip-hop",
            mood="confident",
            bars=8,
        )
        self.assertIn("exp-005 SFT", notes[0])

    def test_exp_005_grpo_model_uses_adapter(self):
        with mock.patch.object(main, "_generate_qwen_verse", return_value="[Verse]\none") as generate:
            lyrics, notes = main._generate_verse_for_model(90, "qwen-exp-005-grpo")

        self.assertEqual(lyrics, "[Verse]\none")
        generate.assert_called_once_with(
            90,
            main.EXP_005_GRPO_ADAPTER,
            genre="Korean hip-hop",
            mood="confident",
            bars=8,
        )
        self.assertIn("exp-005 GRPO", notes[0])

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


def _write_test_wav(path: Path, duration_sec: float = 0.1, sample_rate: int = 8000) -> None:
    frame_count = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count)


if __name__ == "__main__":
    unittest.main()
