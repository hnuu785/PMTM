import unittest

from render import _constrain_ai_phoneme_durations


class RenderDurationTests(unittest.TestCase):
    def test_ai_duration_weights_cannot_steal_phoneme_floors(self):
        durations = _constrain_ai_phoneme_durations(
            ["ch", "e", "ng"],
            0.16,
            [0.0464, 0.0232, 0.0348],
        )

        self.assertAlmostEqual(sum(durations), 0.16)
        self.assertGreaterEqual(durations[0], 0.04)
        self.assertGreaterEqual(durations[1], 0.085)
        self.assertGreaterEqual(durations[2], 0.035)

    def test_infeasible_ai_duration_floors_relax_together(self):
        durations = _constrain_ai_phoneme_durations(
            ["ch", "e", "ng"],
            0.107143,
            [0.0464, 0.0232, 0.0348],
        )

        self.assertAlmostEqual(sum(durations), 0.107143)
        self.assertTrue(all(duration > 0 for duration in durations))
        self.assertGreater(durations[1], 0.05)


if __name__ == "__main__":
    unittest.main()
