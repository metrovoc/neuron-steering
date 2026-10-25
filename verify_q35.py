"""Check that an exported control vector is the same intervention as lyra's neuron push, on Qwen3.5-4B.

Runs her explorer's runtime (the push at down_proj's input) and, on the same loaded model, a forward hook that adds the
exported vectors to the layer outputs, which is what a control vector does. Compares next-token distributions, then
greedy replies. Writes out/<name>.json and the GGUF for llama.cpp.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "neuron-explorer", "explorer"))
import runtime as lyra  # noqa: E402

import ncvec  # noqa: E402

PROMPTS = [
    "Tell me about the moon landing.",
    "Tell me about the pyramids of Egypt.",
    "Is fluoride in tap water safe?",
    "Why do we have daylight saving time?",
    "What did you do this weekend?",
]


def greedy(rt, ids, max_new):
    eos = rt.tok.convert_tokens_to_ids("<|im_end|>")
    out = []
    with torch.inference_mode():
        o = rt.model(input_ids=torch.tensor([ids], device=rt.dev), use_cache=True, logits_to_keep=1)
        for _ in range(max_new):
            t = int(o.logits[0, -1].argmax())
            if t == eos:
                break
            out.append(t)
            o = rt.model(input_ids=torch.tensor([[t]], device=rt.dev), past_key_values=o.past_key_values,
                         use_cache=True, logits_to_keep=1)
    return rt.tok.decode(out, skip_special_tokens=True)


def logprobs(rt, ids):
    with torch.inference_mode():
        o = rt.model(input_ids=torch.tensor([ids], device=rt.dev), use_cache=False, logits_to_keep=1)
    return torch.log_softmax(o.logits[0, -1].float(), -1).cpu()


def kl(p, q):
    return float((p.exp() * (p - q)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", type=ncvec.parse_push, action="append", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-new", type=int, default=120)
    a = ap.parse_args()

    rt = lyra.Runtime(stats_dir=os.path.join(HERE, "out", "no-stats"))  # no sweep stats: her reference-text unit
    rt.ensure_loaded()
    model = os.path.dirname(rt.tok.name_or_path) if os.path.isdir(rt.tok.name_or_path) else rt.tok.name_or_path

    # her unit, from her runtime, against ours, computed independently
    theirs = {l: rt.layer_resid_p50(l) for l, *_ in a.push}
    vectors, ours = ncvec.build(model, a.push)
    for l in theirs:
        print(f"L{l} median resid norm: explorer {theirs[l]:.4f}, ncvec {ours[l]:.4f}")
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    gguf_path = os.path.join(HERE, "out", f"{a.name}.gguf")
    ncvec.write_cvec(gguf_path, vectors, len(rt.lm.layers), "qwen35")

    # the control vector, applied as llama.cpp does: added to the output of layer l
    vec = {l: torch.tensor(v, dtype=torch.bfloat16, device=rt.dev) for l, v in vectors.items()}
    state = {"on": False}

    def add(l):
        def hook(mod, inp, out):
            if not state["on"]:
                return None
            if isinstance(out, tuple):
                return (out[0] + vec[l],) + tuple(out[1:])
            return out + vec[l]
        return hook

    for l in vec:
        rt.lm.layers[l].register_forward_hook(add(l))

    pushes = [(l, n, d, "add") for l, n, d, e in a.push]

    def mode(m):
        rt._push = [rt._push_entry(l, n, d, md, (1,)) for l, n, d, md in pushes] if m == "push" else None
        state["on"] = m == "cvec"

    rows = []
    for p in PROMPTS:
        ids = rt.chat_ids(p)
        lp = {}
        for m in ("base", "push", "cvec"):
            mode(m)
            lp[m] = logprobs(rt, ids)
        replies = {}
        for m in ("base", "push", "cvec"):
            mode(m)
            replies[m] = greedy(rt, ids, a.max_new)
        mode("base")
        r = {"prompt": p, "kl_push_vs_base": kl(lp["push"], lp["base"]), "kl_cvec_vs_push": kl(lp["cvec"], lp["push"]),
             "top1_same": int(lp["push"].argmax()) == int(lp["cvec"].argmax()),
             "replies_identical": replies["push"] == replies["cvec"], **{f"reply_{m}": t for m, t in replies.items()}}
        rows.append(r)
        print(f"\n## {p}\nKL(push||base) {r['kl_push_vs_base']:.3f}  KL(cvec||push) {r['kl_cvec_vs_push']:.2e}  "
              f"identical replies: {r['replies_identical']}")
        for m in ("base", "push", "cvec"):
            print(f"[{m}] {replies[m][:300]!r}")
    json.dump({"pushes": [list(p) for p in a.push], "resid_p50_explorer": theirs, "resid_p50_ncvec": ours,
               "rows": rows}, open(os.path.join(HERE, "out", f"{a.name}.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
