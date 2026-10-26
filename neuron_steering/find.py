"""Find character neurons in Qwen3.5-4B, beyond word search.

    python -m neuron_steering.find concept "QUERY" ...          neuron poles whose steer lens means the query (any language)
    python -m neuron_steering.find contrast "a pirate" ...      neurons a character system prompt moves at the start of the reply
    python -m neuron_steering.find hub L N                      a neuron's strongest direct-path drivers and readers
    python -m neuron_steering.find steer CANDIDATES.json        greedy replies to four open prompts per push stack, ranked

`concept` embeds every token of neuron-explorer's steer lens (the next-token effect of a ±0.8 push, for every
neuron) with Qwen3-Embedding-0.6B and describes each pole by the IDF-weighted mean of its twelve tokens, so tokens
that half the model pushes (作为, 您的, 无法) count for little. The first run builds out/cache/steer-poles.npz.

`contrast` compares down_proj inputs at the last three prompt tokens (where mood and planning neurons gather the
context) with the system prompt "You are X. Stay in character in every reply." against "You are a helpful assistant."
over twelve user turns, scaled by the spread across turns, so it finds neurons the character moves consistently.
Given several characters, it reports what they share (playing any role) first and ranks each one without it.

`steer` ranks stacks by how far the replies move from the unsteered ones (embedding distance), minus repetition,
plus how much more alike the four replies become (a character that carries across prompts). The JSON is a list of
{"name": ..., "pushes": [[layer, neuron, dose], ...]}; `concept` and `contrast` write one with -o.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "vendor", "neuron-explorer", "explorer", "cache")
OUT = os.path.join(ROOT, "out", "cache")
MODEL, EMBED = "Qwen/Qwen3.5-4B", "Qwen/Qwen3-Embedding-0.6B"
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
PROMPTS = ["Tell me about your weekend.", "Write a short poem about the ocean.", "Who are you?",
           "My friend just got a promotion at work. What should I say to them?"]
USERS = ["Tell me about your weekend.", "What do you think of my new haircut?", "How do I bake bread?",
         "What's the capital of France?", "Can you help me write an email to my landlord?", "What is love?",
         "Recommend me a movie.", "Explain how a car engine works.", "I failed my exam today.", "Good morning!",
         "What should I name my cat?", "Is it going to rain tomorrow?"]


def npz(name):
    return dict(np.load(os.path.join(CACHE, name)).items())


_vocab = None


def vocab():
    global _vocab
    if _vocab is None:
        _vocab = json.load(open(os.path.join(CACHE, "vocab.json")))
    return _vocab


def words(ids, k=8):
    return [vocab()[i] for i in ids[:k]]


class Embedder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(EMBED, padding_side="left")
        self.m = AutoModel.from_pretrained(EMBED, dtype=torch.float16).to(DEV).eval()

    def __call__(self, texts, max_length=160, batch=256):
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch):
                b = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length,
                             return_tensors="pt").to(DEV)
                h = self.m(**b).last_hidden_state[:, -1].float()
                out.append(torch.nn.functional.normalize(h, dim=-1).cpu().numpy())
        return np.concatenate(out)


def poles(emb):
    """Per neuron and side, the IDF-weighted, rank-weighted mean embedding of its steer-lens tokens."""
    path = os.path.join(OUT, "steer-poles.npz")
    if os.path.exists(path):
        return dict(np.load(path).items())
    st = npz("steer_index.npz")
    ids, df = np.unique(np.concatenate([st["up"].ravel(), st["down"].ravel()]), return_counts=True)
    print(f"embedding {len(ids)} steer-lens tokens", file=sys.stderr)
    E = emb([vocab()[i].strip() or vocab()[i] for i in ids], max_length=16, batch=512)
    idf = np.log(2 * st["up"].size / 12 / df).astype(np.float32)
    rank = np.linspace(1.5, 0.5, 12, dtype=np.float32)
    out = {}
    for side in ("up", "down"):
        P = np.zeros(st[side].shape[:2] + (E.shape[1],), np.float16)
        for l in range(P.shape[0]):
            pos = np.searchsorted(ids, st[side][l])
            v = np.einsum("nk,nkd->nd", idf[pos] * rank, E[pos])
            P[l] = v / np.linalg.norm(v, axis=-1, keepdims=True)
        out[side] = P
    os.makedirs(OUT, exist_ok=True)
    np.savez(path, **out)
    return out


def concept(a):
    emb = Embedder()
    P, st = poles(emb), npz("steer_index.npz")
    lo, hi = map(int, a.layers.split("-"))
    cands = []
    for q in a.query:
        e = emb([q])[0].astype(np.float16)
        rows = []
        for side, sign in (("up", 1), ("down", -1)):
            s = (P[side] @ e).astype(np.float32)
            s[:lo] = s[hi + 1:] = -1
            for l, n in zip(*np.unravel_index(np.argsort(-s, axis=None)[:a.k], s.shape)):
                rows.append((float(s[l, n]), int(l), int(n), sign))
        rows.sort(reverse=True)
        print(f"\n## {q}")
        for s, l, n, sign in rows[:a.k]:
            side = "up" if sign > 0 else "down"
            print(f"{s:.3f} L{l}N{n}{'+' if sign > 0 else '-'} KL {st[side + '_kl'][l, n]:.2f} {words(st[side][l, n])}")
            cands.append({"name": f"{q}: L{l}N{n}{'+' if sign > 0 else '-'}", "pushes": [[l, n, sign * a.dose]]})
    if a.out:
        json.dump(cands, open(a.out, "w"), indent=1, ensure_ascii=False)


def load_model():
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
    return tok, m, m.model.language_model


def contrast(a):
    tok, m, lm = load_model()
    sc, st = npz("act_scale.npz"), npz("steer_index.npz")
    seen = {}
    for l, layer in enumerate(lm.layers):
        layer.mlp.down_proj.register_forward_pre_hook(
            lambda mod, x, l=l: seen.__setitem__(l, x[0][0, -3:].float().mean(0).cpu().numpy()))

    def acts(system):
        out = []
        for u in USERS:
            ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": u}],
                                          add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                          return_dict=True)["input_ids"]
            with torch.inference_mode():
                lm(input_ids=ids.to(DEV), use_cache=False)
            out.append(np.stack([seen[l] for l in range(len(lm.layers))]))
        return np.stack(out)

    scale = np.maximum(np.maximum(sc["pos"], -sc["neg"]), 1e-3)
    base = acts("You are a helpful assistant.")
    lo, hi = map(int, a.layers.split("-"))
    deltas = {p: acts(f"You are {p}. Stay in character in every reply.") - base for p in a.persona}
    # with several characters, what they all share (playing any role) is reported once and taken out of each
    shared = np.mean(list(deltas.values()), 0) if len(deltas) > 1 else 0
    groups = [("any of these characters", np.mean(list(deltas.values()), 0))] if len(deltas) > 1 else []
    groups += [(p, d - shared) for p, d in deltas.items()]
    cands = []
    for p, d in groups:
        z = d.mean(0) / (d.std(0) + 0.05 * scale)
        z[:lo] = z[hi + 1:] = 0
        print(f"\n## {p}")
        for i in np.argsort(-np.abs(z), axis=None)[:a.k]:
            l, n = map(int, np.unravel_index(i, z.shape))
            sign = 1 if z[l, n] > 0 else -1
            side = "up" if sign > 0 else "down"
            print(f"L{l}N{n}{'+' if sign > 0 else '-'} z {z[l, n]:+.1f} {words(st[side][l, n])}")
            cands.append({"name": f"{p}: L{l}N{n}{'+' if sign > 0 else '-'}", "pushes": [[l, n, sign * a.dose]]})
    if a.out:
        json.dump(cands, open(a.out, "w"), indent=1, ensure_ascii=False)


def hub(a):
    con, st = npz("connections.npz"), npz("steer_index.npz")
    for side, label in (("up", "driven by"), ("down", "writes into")):
        print(f"\n{label}")
        for k in range(a.k):
            l, n = int(con[f"{side}_layer"][a.layer, a.n, k]), int(con[f"{side}_n"][a.layer, a.n, k])
            sign = "+" if con[f"{side}_sign"][a.layer, a.n, k] > 0 else "-"
            print(f"  L{l}N{n} {sign} z {float(con[f'{side}_z'][a.layer, a.n, k]):.0f} "
                  f"+{words(st['up'][l, n], 5)} -{words(st['down'][l, n], 5)}")


def repetition(t):
    w = re.findall(r"\w+", t.lower())
    g = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(g)) / len(g) if g else 0.0


def steer(a):
    """Every stack's replies to PROMPTS, generated together: each batch row carries its own added vectors."""

    from neuron_steering import ncvec
    cfgs = [{"name": "unsteered", "pushes": []}] + json.load(open(a.candidates))
    tok, m, lm = load_model()
    p50 = ncvec.resid_p50(MODEL, range(len(lm.layers)))
    add = {}
    for l, layer in enumerate(lm.layers):
        layer.register_forward_hook(lambda mod, x, o, l=l: None if l not in add else
                                    ((o[0] + add[l], *o[1:]) if isinstance(o, tuple) else o + add[l]))
    D = lm.config.hidden_size
    text = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False) for p in PROMPTS]
    rows = []
    for i in range(0, len(cfgs), a.batch):
        chunk = cfgs[i:i + a.batch]
        add.clear()
        for c_i, c in enumerate(chunk):
            for l, n, dose in c["pushes"]:
                w = lm.layers[l].mlp.down_proj.weight[:, n].float()
                add.setdefault(l, torch.zeros(len(chunk) * len(PROMPTS), 1, D, device=DEV))
                add[l][c_i * len(PROMPTS):(c_i + 1) * len(PROMPTS), 0] += dose * p50[l] * w / w.norm()
        for l in add:
            add[l] = add[l].to(torch.bfloat16)
        b = tok(text * len(chunk), return_tensors="pt", padding=True).to(DEV)
        with torch.inference_mode():
            o = m.generate(**b, max_new_tokens=a.max_tokens, do_sample=False, repetition_penalty=1.1)
        for c_i, c in enumerate(chunk):
            rows.append({**c, "replies": [tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True).strip()
                                          for x in o[c_i * len(PROMPTS):(c_i + 1) * len(PROMPTS)]]})
        print(f"{len(rows)}/{len(cfgs)}", file=sys.stderr, flush=True)
    add.clear()
    del m, lm
    emb = Embedder()
    X = emb([t for r in rows for t in r["replies"]]).reshape(len(rows), len(PROMPTS), -1)
    pair = np.triu_indices(len(PROMPTS), 1)
    alike = lambda x: float((x @ x.T)[pair].mean())
    for r, x in zip(rows, X):
        r["drift"] = 1 - float((x * X[0]).sum(1).mean())
        r["alike"] = alike(x) - alike(X[0])
        r["repetition"] = float(np.mean([repetition(t) for t in r["replies"]]))
        r["score"] = r["drift"] * (1 - 2 * r["repetition"]) + 0.5 * max(r["alike"], 0)
    rows[1:] = sorted(rows[1:], key=lambda r: -r["score"])
    out = a.out or os.path.splitext(a.candidates)[0] + ".steered.json"
    json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)
    for r in rows[:a.show + 1]:
        print(f"\n== {r['name']} {r['pushes']} score {r.get('score', 0):.2f} drift {r.get('drift', 0):.2f} "
              f"alike {r.get('alike', 0):+.2f} repetition {r.get('repetition', 0):.2f}")
        for t in r["replies"]:
            print("  -", t[:200].replace("\n", " "))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("concept")
    c.add_argument("query", nargs="+")
    c.add_argument("-k", type=int, default=12)
    c.add_argument("--layers", default="8-21", help="first-last layer to search (default 8-21: attitudes, not words)")
    c.add_argument("--dose", type=float, default=1.0)
    c.add_argument("-o", "--out")
    p = sub.add_parser("contrast")
    p.add_argument("persona", nargs="+", help='completes "You are ...", e.g. "a pirate"')
    p.add_argument("-k", type=int, default=12)
    p.add_argument("--layers", default="4-23")
    p.add_argument("--dose", type=float, default=1.0)
    p.add_argument("-o", "--out")
    h = sub.add_parser("hub")
    h.add_argument("layer", type=int)
    h.add_argument("n", type=int)
    h.add_argument("-k", type=int, default=16)
    s = sub.add_parser("steer")
    s.add_argument("candidates")
    s.add_argument("--batch", type=int, default=8, help="stacks generated together")
    s.add_argument("--max-tokens", type=int, default=80)
    s.add_argument("--show", type=int, default=20)
    s.add_argument("-o", "--out")
    a = ap.parse_args()
    {"concept": concept, "contrast": contrast, "hub": hub, "steer": steer}[a.cmd](a)


if __name__ == "__main__":
    main()
