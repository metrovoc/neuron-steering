import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

from neuron_steering import find
from neuron_steering.find import repetition


class RepetitionTests(unittest.TestCase):
    def test_fresh_text_has_no_repetition(self):
        self.assertEqual(repetition("the quick brown fox jumps over the lazy dog"), 0)

    def test_loop_is_mostly_repetition(self):
        self.assertGreater(repetition("I am a sense of wonder. " * 6), 0.7)

    def test_empty_or_letterless_reply_is_degenerate(self):
        self.assertEqual(repetition(""), 1)
        self.assertEqual(repetition("1111111111111111 2222222222"), 1)


class SweepTests(unittest.TestCase):
    def test_resumes_where_it_stopped_and_ranks_against_unsteered(self):
        asked, ranked = [], []

        class Flash:
            def replies(self, cfgs, prompts, max_tokens):
                asked.extend(c["name"] for c in cfgs)
                return [{**c, "replies": ["reply"]} for c in cfgs]

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "sweep.jsonl")
            with open(out, "w") as f:
                f.write(json.dumps({"name": "L20E344N0+", "pushes": [[20, 0, 6.0, "344"]], "replies": ["x"]}) + "\n")
            with mock.patch.object(find, "Flash", Flash), mock.patch.object(find, "rank", lambda rows, *a: ranked.extend(rows)):
                find.sweep(Namespace(experts=["20:344"], dose=6.0, out=out, show=0, toward=None))
            with open(out) as f:
                lines = len(f.readlines())
        self.assertEqual(asked[0], "unsteered")
        self.assertNotIn("L20E344N0+", asked)
        self.assertEqual(len(asked), 1 + 640 * 2 - 1)
        self.assertEqual(lines, 1 + 640 * 2)
        self.assertEqual(ranked[0]["name"], "unsteered")
        self.assertEqual(ranked[1]["pushes"], [[20, 0, 6.0, "344"]])
