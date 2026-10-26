import unittest

from neuron_steering.find import repetition


class RepetitionTests(unittest.TestCase):
    def test_fresh_text_has_no_repetition(self):
        self.assertEqual(repetition("the quick brown fox jumps over the lazy dog"), 0)

    def test_loop_is_mostly_repetition(self):
        self.assertGreater(repetition("I am a sense of wonder. " * 6), 0.7)

    def test_empty_or_letterless_reply_is_degenerate(self):
        self.assertEqual(repetition(""), 1)
        self.assertEqual(repetition("1111111111111111 2222222222"), 1)
