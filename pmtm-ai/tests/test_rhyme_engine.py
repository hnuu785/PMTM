import unittest

from app.rhyme_scoring.rhyme_engine import get_line_rhyme_score


class LineRhymeScoreTests(unittest.TestCase):
    def test_lexically_distinct_same_final_syllable_can_rhyme(self):
        self.assertGreaterEqual(get_line_rhyme_score("빨라", "올라"), 0.5)

    def test_repeated_grammatical_endings_do_not_rhyme(self):
        for left, right in [
            ("없네", "있네"),
            ("했어요", "그랬구요"),
            ("했습니다", "말했습니다"),
        ]:
            with self.subTest(left=left, right=right):
                self.assertEqual(get_line_rhyme_score(left, right), 0.2)

