import unittest

from neuron_steering.find import repetition


class RepetitionTests(unittest.TestCase):
    def test_fresh_text_has_no_repetition(self):
        self.assertEqual(repetition("the quick brown fox jumps over the lazy dog"), 0)

    def test_loop_is_mostly_repetition(self):
        self.assertGreater(repetition("I am a sense of wonder. " * 6), 0.7)

    def test_short_text(self):
        self.assertEqual(repetition("hi"), 0)
