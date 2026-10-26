"""The catalog of neurons worth pushing, `presets.json`: the GUI's example stacks and `docs/neurons.md`.

    uv run python -m neuron_steering.presets      # rewrite docs/neurons.md after editing presets.json
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH, DOC = os.path.join(ROOT, "presets.json"), os.path.join(ROOT, "docs", "neurons.md")
MODELS = {"small": ("Qwen3.5-4B", "doses in neuron-explorer's unit"),
          "flash": ("Qwen3.8-Flash-Next", "doses in the same unit; routed experts by number, the shared expert as \"shared\"")}


def load():
    return json.load(open(PATH))


def push(p):
    layer, neuron, dose, *rest = p
    expert = rest[0] if rest else None
    where = "" if len(p) < 4 else f" expert {expert}" if expert is not None else " shared"
    return f"L{layer}{where} N{neuron} {'+' if dose > 0 else '−'}{abs(dose):g}"


def render(presets):
    out = ["# Neurons", "",
           "Every stack here is in the GUI's stack menu under Examples. They are generated from",
           "[`presets.json`](../presets.json): edit that file, then run `uv run python -m neuron_steering.presets`.",
           "Effects are from greedy replies; how each was found is in [finding-neurons.md](finding-neurons.md) and",
           "[control-vectors.md](control-vectors.md)."]
    for model, entries in presets.items():
        label, unit = MODELS[model]
        out += ["", f"## {label}", "", f"Pushes as layer, neuron and dose ({unit}).", "",
                "| Preset | Pushes | Found by | Effect |", "| --- | --- | --- | --- |"]
        out += [f"| {e['name']} | {', '.join(push(p) for p in e['pushes'])} | {e['found']} | {e['effect']} |"
                for e in entries]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    open(DOC, "w").write(render(load()))
    print(DOC)
