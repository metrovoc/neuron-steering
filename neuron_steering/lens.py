"""Logit-lens search over a model's MLP neurons from the weights alone: each neuron's down column through the final norm
and the unembedding, top tokens at both poles, filtered by a regex (lyra's explorer's `lens` search, without a model run).

    python -m neuron_steering.lens --model Qwen/Qwen3.8-Flash-Next --expert shared --layers 8-44 --match 'conspir|secret|illumin'

Writes out/lens-<model>-<expert>.json: every neuron's top tokens, for later searches without the weights.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from transformers import AutoTokenizer

from neuron_steering import ncvec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def final_norm_weight(w, hidden):
    """The final RMSNorm's weight; with hyper-connections the mixer's per-stream norms, averaged over the streams."""
    names = w.names()
    for n in names:
        if n.endswith("language_model.norm.weight") or n == "model.norm.weight":
            return w.tensor(n)
    n = next(n for n in names if n.endswith("language_model.hyper_connection_mixer.hc_norm.weight"))
    return w.tensor(n).reshape(-1, hidden).mean(0)


def scan_routed(a, tok, rx):
    """Every routed expert's neurons, a layer at a time from HTTP range reads; keeps only the neurons with at least two
    matching tokens among a pole's top k (the full lens of 300k neurons a layer would not fit)."""
    out_path = os.path.join(ROOT, "out", f"lens-{a.model.replace('/', '_')}-routed.jsonl")
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(s)["layer"] for s in open(out_path) if '"layer_done"' in s}
    w = ncvec.Weights(a.model)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    WU = torch.tensor(w.tensor("lm_head.weight"), dtype=torch.bfloat16, device=dev)
    g = torch.tensor(final_norm_weight(w, WU.shape[1]), device=dev)
    vocab = [tok.convert_tokens_to_string([t]) for t in tok.convert_ids_to_tokens(list(range(len(tok))))]
    vocab += [""] * (WU.shape[0] - len(vocab))  # padded unembedding rows
    concept = torch.tensor([bool(rx.search(t)) for t in vocab], device=dev)
    lo, hi = map(int, a.layers.split("-"))
    with open(out_path, "a") as f:
        for l in range(lo, hi + 1):
            if l in done:
                continue
            name = ncvec.down_name(w, l, 0)
            n_exp = 512
            for e0 in range(0, n_exp, 16):
                D = torch.tensor(w.tensor(name, rows=(e0, e0 + 16)), device=dev)  # [16, hidden, inter]
                cols = D.permute(0, 2, 1).reshape(-1, D.shape[1])  # [16 * inter, hidden]
                x = (cols / cols.norm(dim=-1, keepdim=True) * g).to(torch.bfloat16)
                for c0 in range(0, x.shape[0], 2048):
                    logits = (x[c0:c0 + 2048] @ WU.T).float()
                    for pole, lg in (("up", logits), ("down", -logits)):
                        top = lg.topk(a.k, -1).indices
                        hits = concept[top].sum(-1)
                        for i in (hits >= 2).nonzero().flatten().tolist():
                            j = c0 + i
                            rec = {"layer": l, "expert": e0 + j // D.shape[2], "neuron": j % D.shape[2], "pole": pole,
                                   "hits": int(hits[i]), "top": [vocab[t] for t in top[i, :12].tolist()]}
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.write(json.dumps({"layer": l, "layer_done": True}) + "\n")
            f.flush()
            print(f"layer {l} done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--expert", help="'shared', an expert number, or 'all' routed experts (needs --match; default: the dense MLP)")
    ap.add_argument("--layers", required=True, help="a-b, inclusive")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--match", help="regex over the top tokens; prints the neurons that hit")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.expert == "all":
        return scan_routed(a, tok, re.compile(a.match, re.I))
    out_path = os.path.join(ROOT, "out", f"lens-{a.model.replace('/', '_')}-{a.expert or 'dense'}.json")
    lens = json.load(open(out_path)) if os.path.exists(out_path) else {}
    lo, hi = map(int, a.layers.split("-"))
    todo = [l for l in range(lo, hi + 1) if str(l) not in lens]
    if todo:
        w = ncvec.Weights(a.model)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        WU = torch.tensor(w.tensor("lm_head.weight"), dtype=torch.bfloat16, device=dev)
        g = torch.tensor(final_norm_weight(w, WU.shape[1]), device=dev)
        vocab = tok.convert_ids_to_tokens(list(range(len(tok))))
        for l in todo:
            name = ncvec.down_name(w, l, a.expert)
            D = w.tensor(name, rows=(int(a.expert), int(a.expert) + 1))[0] if name.endswith("experts.down_proj") else w.tensor(name)
            cols = torch.tensor(D, device=dev).T  # [neurons, hidden]
            x = cols / cols.norm(dim=-1, keepdim=True) * g
            logits = (x.to(torch.bfloat16) @ WU.T).float()  # [neurons, vocab]
            logits = logits[:, : len(vocab)]
            up = logits.topk(a.k, -1).indices.cpu().numpy()
            down = (-logits).topk(a.k, -1).indices.cpu().numpy()
            lens[str(l)] = [{"up": [tok.convert_tokens_to_string([vocab[i]]) for i in up[n]],
                             "down": [tok.convert_tokens_to_string([vocab[i]]) for i in down[n]]} for n in range(len(up))]
            print(f"layer {l}: {len(up)} neurons", flush=True)
            json.dump(lens, open(out_path, "w"), ensure_ascii=False)
    if a.match:
        rx = re.compile(a.match, re.I)
        for l in range(lo, hi + 1):
            for n, e in enumerate(lens[str(l)]):
                for pole in ("up", "down"):
                    hits = [t for t in e[pole] if rx.search(t)]
                    if len(hits) >= 2:
                        print(f"L{l} N{n} {pole:4s} {len(hits)} | {' '.join(t.strip() for t in e[pole][:12])}")


if __name__ == "__main__":
    main()
