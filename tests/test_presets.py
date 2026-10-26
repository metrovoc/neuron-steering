import unittest

from neuron_steering import presets


class PresetTests(unittest.TestCase):
    def test_doc_is_current(self):
        self.assertEqual(open(presets.DOC).read(), presets.render(presets.load()),
                         "run uv run python -m neuron_steering.presets")

    def test_pushes_name_an_expert_only_on_flash(self):
        for model, entries in presets.load().items():
            for e in entries:
                for p in e["pushes"]:
                    self.assertEqual(len(p), 4 if model == "flash" else 3, e["name"])


if __name__ == "__main__":
    unittest.main()
