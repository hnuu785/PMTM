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
        # 120 BPM is scaled to 60 BPM (half-time), so bar duration is 4.0s.
        # Last bar (index 7) startSec = 1.25 + 7 * 4.0 = 29.25
        self.assertEqual(plan.bars[-1].startSec, 29.25)
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

    def test_flow_plan_supports_english_lyrics(self):
        lyrics = EIGHT_BARS.replace("나는 비트 위를 달려", "I'm on my way to the future")
        plan = build_flow_plan(lyrics, 90, 0, "potg", base_f0_hz=190.0)
        self.assertEqual(len(plan.bars), 8)
        self.assertIn("ay", [p.symbol for p in plan.bars[0].phonemes])

    def test_flow_plan_rejects_unsupported_characters(self):
        lyrics = EIGHT_BARS.replace("나는 비트 위를 달려", "나는 🚀 위를 달려")
        with self.assertRaisesRegex(ValueError, "지원하지 않는 문자"):
            build_flow_plan(lyrics, 90, 0, "potg", base_f0_hz=190.0)


    def test_flow_plan_supports_up_to_24_syllables(self):
        # 24 syllables should compile successfully and use the 24-syllable template
        dense_lyrics = "\n".join([
            "가나다라마바사아자차카타파하가나다라마바사아자",  # 24 syllables
            "고개를 들고 앞을 봐",
            "작은 불씨 크게 번져",
            "흔들림 없이 길을 가",
            "다시 박자 위에 올라",
            "숨을 고르고 말을 해",
            "오늘보다 멀리 날아",
            "마지막까지 나를 믿어",
        ])
        plan = build_flow_plan(dense_lyrics, 90, 0, "potg", base_f0_hz=190.0)
        self.assertEqual(len(plan.bars), 8)

    def test_flow_plan_supports_7_syllables(self):
        # 7 syllables should compile successfully and use the 7-syllable template
        sparse_lyrics = "\n".join([
            "가나다라마바사",  # 7 syllables
            "고개를 들고 앞을 봐",
            "작은 불씨 크게 번져",
            "흔들림 없이 길을 가",
            "다시 박자 위에 올라",
            "숨을 고르고 말을 해",
            "오늘보다 멀리 날아",
            "마지막까지 나를 믿어",
        ])
        plan = build_flow_plan(sparse_lyrics, 90, 0, "potg", base_f0_hz=190.0)
        self.assertEqual(len(plan.bars), 8)

    def test_hierarchical_word_rhythm_allocation(self):
        # Test word-chunk based hierarchical rhythm allocation with micro-pauses
        test_lyrics = "\n".join([
            "그걸 보고 감동하는 너에게 감동",  # 5 words, 13 syllables
            "고개를 들고 앞을 봐",
            "작은 불씨 크게 번져",
            "흔들림 없이 길을 가",
            "다시 박자 위에 올라",
            "숨을 고르고 말을 해",
            "오늘보다 멀리 날아",
            "마지막까지 나를 믿어",
        ])
        plan = build_flow_plan(test_lyrics, 90, 0, "potg", base_f0_hz=190.0)
        bar0 = plan.bars[0]
        self.assertIn("adaptive_hierarchical_boom_bap_13syl_5words", bar0.template)
        # Check that inter-word SPs are removed for legato flow (only lead & tail SP remain)
        sp_count = sum(1 for p in bar0.phonemes if p.symbol == "SP")
        self.assertEqual(sp_count, 2)
        self.assertAlmostEqual(
            sum(p.durationSec for p in bar0.phonemes),
            plan.beatMap.barDurationSec,
            places=5,
        )

    def test_adaptive_min_dur_and_max_cap(self):
        # Test 13-syllable bar at 130 BPM ensures minimum phoneme duration is guaranteed
        dense_130bpm = "\n".join([
            "쏟아지는 빗속에서 기다려본 적?",  # 13 syllables at 130 BPM
            "고개를 들고 앞을 봐",
            "작은 불씨 크게 번져",
            "흔들림 없이 길을 가",
            "다시 박자 위에 올라",
            "숨을 고르고 말을 해",
            "오늘보다 멀리 날아",
            "마지막까지 나를 믿어",
        ])
        plan = build_flow_plan(dense_130bpm, 130, 0, "potg", base_f0_hz=190.0)
        bar0 = plan.bars[0]
        self.assertIn("13syl", bar0.template)
        # Ensure total duration matches bar duration
        self.assertAlmostEqual(
            sum(p.durationSec for p in bar0.phonemes),
            plan.beatMap.barDurationSec,
            places=5,
        )

        # Test single-syllable bar enforces MAX_SYLLABLE_DUR_SEC (0.60s) Cap
        sparse_lyrics = "\n".join([
            "아",  # 1 syllable
            "고개를 들고 앞을 봐",
            "작은 불씨 크게 번져",
            "흔들림 없이 길을 가",
            "다시 박자 위에 올라",
            "숨을 고르고 말을 해",
            "오늘보다 멀리 날아",
            "마지막까지 나를 믿어",
        ])
        plan_sparse = build_flow_plan(sparse_lyrics, 90, 0, "potg", base_f0_hz=190.0)
        bar0_sparse = plan_sparse.bars[0]
        # Syllable phonemes (for '아' -> vowel 'a') should not exceed 0.60s
        syllable_dur = sum(p.durationSec for p in bar0_sparse.phonemes if p.symbol != "SP")
        self.assertLessEqual(syllable_dur, 0.600001)

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

    def test_kiwi_morpheme_stress_and_dynamic_pitch_cadence(self):
        plan = build_flow_plan(EIGHT_BARS, 90, 0, "potg", base_f0_hz=190.0)
        bar0 = plan.bars[0]
        self.assertIsNotNone(bar0.noteSeq)
        self.assertIn("rest", bar0.noteSeq)
        # Check linguistic dynamic pitch notes are assigned
        self.assertTrue(any(note in bar0.noteSeq for note in ("D4", "D#4", "C4")))


    def test_genre_boom_bap_vs_trap_timing_modulation(self):
        plan_bb = build_flow_plan(EIGHT_BARS, 90, 0, "potg", base_f0_hz=190.0, genre="boom_bap")
        plan_trap = build_flow_plan(EIGHT_BARS, 90, 0, "potg", base_f0_hz=190.0, genre="trap")
        self.assertEqual(plan_bb.genre, "boom_bap")
        self.assertEqual(plan_trap.genre, "trap")
        # Check template names reflect the genre
        self.assertIn("boom_bap", plan_bb.bars[0].template)
        self.assertIn("trap", plan_trap.bars[0].template)
        # Compare phoneme durations between boom_bap (layback) and trap (early push)
        dur_bb = [p.durationSec for p in plan_bb.bars[0].phonemes]
        dur_trap = [p.durationSec for p in plan_trap.bars[0].phonemes]
        self.assertNotEqual(dur_bb, dur_trap)

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
