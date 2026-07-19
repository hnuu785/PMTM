import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient

from app import main
from app import guide_pipeline
from app.flow_adapter import (
    _syllable_to_phonemes,
    build_flow_plan,
    parse_eight_bar_lyrics,
    write_diffsinger_ds,
)


EIGHT_BARS = "\n".join(
    [
        "나는 비트 위를 달려",
        "고개를 들고 앞을 봐",
        "작은 불씨 크게 번져",
        "흔들림 없이 길을 가",
        "다시 박자 위에 올라",
        "숨을 고르고 말을 해",
        "오늘보다 멀리 날아",
        "마지막까지 나를 믿어",
    ]
)


class FlowAdapterTests(unittest.TestCase):
    def test_korean_phonemes_match_potg_inventory(self):
        self.assertEqual(_syllable_to_phonemes("각"), ["g", "a", "kcl"])
        self.assertEqual(_syllable_to_phonemes("시"), ["sh", "i"])
        self.assertEqual(_syllable_to_phonemes("워"), ["w", "o"])
        self.assertEqual(_syllable_to_phonemes("의"), ["ui"])

    def test_korean_liaison_g2p_rules(self):
        plan = build_flow_plan(
            "\n".join(["접어"] * 8),
            120,
            0.0,
            "potg",
            base_f0_hz=190.0,
        )
        self.assertEqual(plan.bars[0].text, "저버")
        symbols = [p.symbol for p in plan.bars[0].phonemes]
        self.assertEqual(symbols, ["SP", "jh", "eo", "b", "eo", "SP"])

    def test_flow_plan_is_exactly_eight_aligned_bars(self):
        plan = build_flow_plan(EIGHT_BARS, 120, 1.25, "potg", base_f0_hz=190.0)

        self.assertEqual(plan.beatMap.barCount, 8)
        self.assertEqual(len(plan.bars), 8)
        self.assertEqual(plan.bars[0].startSec, 1.25)
        self.assertEqual(plan.bars[-1].startSec, 15.25)
        for bar in plan.bars:
            self.assertAlmostEqual(
                sum(phoneme.durationSec for phoneme in bar.phonemes),
                plan.beatMap.barDurationSec,
                places=5,
            )
            self.assertEqual(bar.phonemes[0].symbol, "SP")
            self.assertEqual(bar.phonemes[-1].symbol, "SP")

    def test_flow_plan_rejects_non_eight_line_lyrics(self):
        with self.assertRaisesRegex(ValueError, "정확히 8줄"):
            parse_eight_bar_lyrics("한 줄\n두 줄")

    def test_flow_plan_rejects_unsupported_english(self):
        lyrics = EIGHT_BARS.replace("나는 비트 위를 달려", "나는 beat 위를 달려")
        with self.assertRaisesRegex(ValueError, "한글 가사만 지원"):
            build_flow_plan(lyrics, 90, 0, "potg", base_f0_hz=190.0)

    def test_diffsinger_score_contains_eight_manual_sections(self):
        plan = build_flow_plan(EIGHT_BARS, 90, 2.0, "rang", base_f0_hz=145.0)
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "score.ds"
            write_diffsinger_ds(plan, output, base_f0_hz=145.0)
            sections = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(sections), 8)
        self.assertEqual(sections[0]["offset"], "2.000000")
        self.assertIn("ph_dur", sections[0])
        self.assertIn("f0_seq", sections[0])
        self.assertEqual(len(sections[0]["ph_seq"].split()), len(sections[0]["ph_dur"].split()))
        self.assertEqual(
            sum(int(value) for value in sections[0]["ph_num"].split()),
            len(sections[0]["ph_seq"].split()),
        )
        self.assertLess(
            len(sections[0]["ph_num"].split()),
            len(sections[0]["ph_seq"].split()),
        )
        self.assertEqual(set(sections[0]["energy"].split()), {"-80.0", "-26.0"})
        self.assertEqual(set(sections[0]["breathiness"].split()), {"-80.0", "-60.0"})
        self.assertEqual(
            len(sections[0]["f0_seq"].split()),
            len(sections[0]["energy"].split()),
        )

    def test_f0_curve_is_flat(self):
        from app.flow_adapter import _build_f0_curve
        symbols = ["SP", "g", "a", "kcl", "SP"]
        durations = [0.1, 0.2, 0.3, 0.2, 0.1]
        base_f0 = 190.0
        f0_curve = _build_f0_curve(symbols, durations, base_f0, 1)
        
        # Total duration = 0.9s, timestep = 0.01s => 90 frames
        self.assertEqual(len(f0_curve), 90)
        
        # SP frames (0.1s => 10 frames) should be 0.0
        for val in f0_curve[:10]:
            self.assertEqual(val, 0.0)
            
        for val in f0_curve[-10:]:
            self.assertEqual(val, 0.0)
            
        # Voiced frames (g, a, kcl) should be > 0.0 and <= base_f0
        voiced_f0s = f0_curve[10:-10]
        self.assertTrue(all(val > 0.0 for val in voiced_f0s))
        self.assertAlmostEqual(max(voiced_f0s), base_f0)
        
        # Check if F0 is indeed flat (most values equal base_f0, except near boundary ramps)
        # 70 frames of voiced segment. Glide frames = min(4, 70 // 2) = 4 frames on each side.
        # So frames 14 to 76 (relative index 4 to 66) should be exactly base_f0.
        for idx, val in enumerate(voiced_f0s):
            if 4 <= idx < len(voiced_f0s) - 4:
                self.assertAlmostEqual(val, base_f0)


class GuideDemoApiTests(unittest.TestCase):
    def test_guide_demo_enqueues_edited_eight_lines(self):
        client = TestClient(main.app)
        redis_client = mock.Mock()
        with (
            TemporaryDirectory() as storage_dir,
            mock.patch.object(main, "DEMO_STORAGE_ROOT", Path(storage_dir)),
            mock.patch.object(main, "_get_redis_client", return_value=redis_client),
            mock.patch.object(main, "enqueue_guide_demo") as enqueue,
        ):
            response = client.post(
                "/api/v1/guide-demos",
                data={
                    "lyrics": EIGHT_BARS,
                    "bpm": "92",
                    "firstBarStartSec": "3.25",
                    "voicebank": "rang",
                },
                files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "queued")
        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[4], EIGHT_BARS)
        self.assertEqual(enqueue.call_args.args[5], 92)
        self.assertEqual(enqueue.call_args.args[6], 3.25)
        self.assertEqual(enqueue.call_args.args[7], "rang")

    def test_guide_demo_rejects_seven_lines(self):
        client = TestClient(main.app)
        response = client.post(
            "/api/v1/guide-demos",
            data={
                "lyrics": "\n".join(EIGHT_BARS.splitlines()[:7]),
                "bpm": "92",
                "firstBarStartSec": "0",
                "voicebank": "potg",
            },
            files={"beat": ("beat.wav", b"audio bytes", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("정확히 8줄", response.json()["detail"])

    def test_demo_status_includes_svs_artifacts(self):
        client = TestClient(main.app)
        redis_client = mock.Mock()
        redis_client.hgetall.return_value = {
            "jobId": "abc",
            "status": "succeeded",
            "progress": "1.0",
            "bpm": "92",
            "voicebank": "potg",
            "audioUrl": "/api/v1/demos/abc/audio",
            "vocalUrl": "/api/v1/demos/abc/vocal",
            "flowPlanUrl": "/api/v1/demos/abc/flow-plan",
        }

        with mock.patch.object(main, "_get_redis_client", return_value=redis_client):
            response = client.get("/api/v1/demos/abc")

        body = response.json()
        self.assertEqual(body["voicebank"], "potg")
        self.assertEqual(body["vocalUrl"], "/api/v1/demos/abc/vocal")
        self.assertEqual(body["flowPlanUrl"], "/api/v1/demos/abc/flow-plan")

    def test_worker_writes_flow_score_vocal_and_mix_for_eight_bars(self):
        class FakeRedisClient:
            def __init__(self):
                self.payload = {}

            def hset(self, _key, mapping):
                self.payload.update(mapping)

            def expire(self, *_args):
                pass

        fake_redis_client = FakeRedisClient()
        fake_redis_module = mock.Mock(Redis=mock.Mock(from_url=mock.Mock(return_value=fake_redis_client)))

        with (
            TemporaryDirectory() as tmp,
            mock.patch.dict("sys.modules", {"redis": fake_redis_module}),
            mock.patch.object(guide_pipeline, "_trim_beat") as trim,
            mock.patch.object(guide_pipeline, "_fit_vocal_to_duration") as fit_vocal,
            mock.patch.object(guide_pipeline, "render_diffsinger") as render,
            mock.patch.object(guide_pipeline, "mix_demo_audio") as mix,
        ):
            work_dir = Path(tmp)
            beat_path = work_dir / "input.wav"
            beat_path.write_bytes(b"beat")
            trim.side_effect = lambda _input, output, _duration: output.write_bytes(b"beat")
            render.side_effect = lambda _score, output, _voicebank: output.write_bytes(b"RIFFraw")
            fit_vocal.side_effect = lambda _input, output, _duration: output.write_bytes(b"RIFFvocal")
            mix.side_effect = lambda _beat, _vocal, root: (root / "demo.mp3")

            guide_pipeline.run_guide_demo_generation(
                "job",
                str(beat_path),
                str(work_dir),
                EIGHT_BARS,
                90,
                1.0,
                "potg",
            )

            self.assertTrue((work_dir / "flow-plan.json").is_file())
            self.assertTrue((work_dir / "score.ds").is_file())
            self.assertTrue((work_dir / "vocal.wav").is_file())

        self.assertEqual(fake_redis_client.payload["status"], "succeeded")
        self.assertEqual(fake_redis_client.payload["voicebank"], "potg")


if __name__ == "__main__":
    unittest.main()
