"""Find character neurons in Qwen3.5-4B (and, for contrast and steer, Qwen3.8-Flash-Next), beyond word search.

    python -m neuron_steering.find concept "QUERY" ...          neuron poles whose steer lens means the query (any language)
    python -m neuron_steering.find contrast "a pirate" ...      neurons a character system prompt moves at the start of the reply
    python -m neuron_steering.find hub L N                      a neuron's strongest direct-path drivers and readers
    python -m neuron_steering.find steer CANDIDATES.json        greedy replies to four open prompts per push stack, ranked
    python -m neuron_steering.find sweep 32:480 -o OUT.jsonl    every neuron of a Flash-Next expert at ±6, one reply each

`concept` embeds every token of neuron-explorer's steer lens (the next-token effect of a ±0.8 push, for every
neuron) with Qwen3-Embedding-0.6B and describes each pole by the IDF-weighted mean of its twelve tokens, so tokens
that half the model pushes (作为, 您的, 无法) count for little. The first run builds out/cache/steer-poles.npz.

`contrast` compares down_proj inputs at the last three prompt tokens (where mood and planning neurons gather the
context) with the system prompt "You are X. Stay in character in every reply." against "You are a helpful assistant."
over twelve user turns, scaled by the spread across turns, so it finds neurons the character moves consistently.
Given several characters, it reports what they share (playing any role) first and ranks each one without it.
`--model flash` reads Qwen3.8-Flash-Next through the GUI's Strata server (STRATA_METAL_ACTS) and ranks the shared
expert's neurons and those of routed experts the character recruits; `steer --model flash` steers through
STRATA_METAL_CVEC_LIVE. Flash-Next pushes are [layer, neuron, dose, expert] (expert null for the shared expert).

`steer` ranks stacks by how far the replies move from the unsteered ones (embedding distance), minus repetition,
plus how much more alike its replies become (a character that carries across prompts); with --toward, by how
far they move in the direction a reference stack moved its model (to look for a known behaviour in another model).
--wide adds four prompts asking for advice and sympathy, where attitudes to the user show. The JSON is a list of
{"name": ..., "pushes": [[layer, neuron, dose], ...]}; `concept` and `contrast` write one with -o.

`sweep` reads what one routed expert holds without a hypothesis: each neuron pushed alone, both signs, with one short
reply to the first prompt (about 45 stacks a minute). It appends to OUT.jsonl, so an interrupted sweep resumes, and
ranks like `steer`.
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
WIDE = ["My dog is sick, what should I do?", "Give me advice for my first date.", "I'm feeling lonely today.",
        "What do you think about humans?"]
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


class Small4B:
    """Qwen3.5-4B in transformers: activations at the reply's start, and batched steering with per-row vectors."""
    name, dose = "Qwen3.5-4B", 1.0

    def __init__(self):
        self.tok, self.m, self.lm = load_model()
        self.st = npz("steer_index.npz")
        sc = npz("act_scale.npz")
        self.scale = np.maximum(np.maximum(sc["pos"], -sc["neg"]), 1e-3)  # [L, N]
        self.seen = None  # a dict while acts() reads a prompt
        for l, layer in enumerate(self.lm.layers):
            layer.mlp.down_proj.register_forward_pre_hook(lambda mod, x, l=l: None if self.seen is None else
                                                          self.seen.__setitem__(l, x[0][0, -3:].float().mean(0).cpu().numpy()))

    def acts(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False,
                                           return_tensors="pt", return_dict=True)["input_ids"]
        self.seen = {}
        with torch.inference_mode():
            self.lm(input_ids=ids.to(DEV), use_cache=False)
        x, self.seen = np.stack([self.seen[l] for l in range(len(self.lm.layers))]), None
        return x

    def describe(self, idx, sign):
        l, n = idx
        return [int(l), int(n)], f"L{l}N{n}", words(self.st["up" if sign > 0 else "down"][l, n])

    def replies(self, cfgs, prompts, max_tokens, batch):
        """Every stack's replies, generated together: each batch row carries its own added vectors."""
        from neuron_steering import ncvec
        tok, m, lm = self.tok, self.m, self.lm
        p50 = ncvec.resid_p50(MODEL, range(len(lm.layers)))
        add = {}
        hooks = [layer.register_forward_hook(lambda mod, x, o, l=l: None if l not in add else
                                             ((o[0] + add[l], *o[1:]) if isinstance(o, tuple) else o + add[l]))
                 for l, layer in enumerate(lm.layers)]
        D = lm.config.hidden_size
        text = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False) for p in prompts]
        rows = []
        for i in range(0, len(cfgs), batch):
            chunk = cfgs[i:i + batch]
            add.clear()
            for c_i, c in enumerate(chunk):
                for l, n, dose in c["pushes"]:
                    w = lm.layers[l].mlp.down_proj.weight[:, n].float()
                    add.setdefault(l, torch.zeros(len(chunk) * len(prompts), 1, D, device=DEV))
                    add[l][c_i * len(prompts):(c_i + 1) * len(prompts), 0] += dose * p50[l] * w / w.norm()
            for l in add:
                add[l] = add[l].to(torch.bfloat16)
            b = tok(text * len(chunk), return_tensors="pt", padding=True).to(DEV)
            with torch.inference_mode():
                o = m.generate(**b, max_new_tokens=max_tokens, do_sample=False, repetition_penalty=1.1)
            for c_i, c in enumerate(chunk):
                rows.append({**c, "replies": [tok.decode(x[b.input_ids.shape[1]:], skip_special_tokens=True).strip()
                                              for x in o[c_i * len(prompts):(c_i + 1) * len(prompts)]]})
            print(f"{len(rows)}/{len(cfgs)}", file=sys.stderr, flush=True)
        for h in hooks:
            h.remove()
        del self.m, self.lm
        return rows


class Flash:
    """Qwen3.8-Flash-Next in the GUI's Strata server (port 8090, STRATA_METAL_ACTS and STRATA_METAL_CVEC_LIVE),
    started if it is not running. Units are [layer, expert, neuron] with expert 512 the shared expert."""
    name, dose = "Qwen3.8-Flash-Next", 6.0

    def __init__(self):
        sys.path.insert(0, ROOT)
        from app.models import Flash as Backend
        from neuron_steering import acts
        self.fl, self.capture = Backend(), acts.capture
        self.fl.ensure()
        self.scale = self.fl.scale()  # [48, 513, 640]

    def acts(self, messages):
        from app.models import STRATA_PORT
        tok = self.fl.tokenizer()
        T = len(tok(tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False),
                    add_special_tokens=False).input_ids)
        with self.fl.lock:
            cap = self.capture(STRATA_PORT, self.fl.acts_file, messages)
        ps = [p for p in range(T - 3, T) if p in cap.ids]
        if not ps:
            raise RuntimeError("the engine reused its cache for the reply's start; nothing was captured")
        x = np.zeros((48, 513, 640), np.float32)
        self.routed = np.zeros((48, 513), np.float32)
        self.routed[:, 512] = 1
        layers = np.arange(48)[:, None]
        for p in ps:
            x[:, 512] += cap.hs[p]
            x[layers, cap.ids[p]] += cap.h[p]
            self.routed[layers, cap.ids[p]] += 1 / len(ps)
        return x / len(ps)

    def describe(self, idx, sign):
        l, e, n = map(int, idx)
        expert = None if e == 512 else str(e)
        lens = self.fl.lens(l, n, expert, k=8)["up" if sign > 0 else "down"]
        return [l, n, expert], f"L{l}{'S' if expert is None else 'E' + expert}N{n}", lens

    def replies(self, cfgs, prompts, max_tokens):
        """One live vector per stack; the GUI's vector file is put back afterwards."""
        live = open(self.fl.live, "rb").read() if os.path.exists(self.fl.live) else None
        rows = []
        try:
            for c in cfgs:
                pushes = [{"layer": l, "neuron": n, "dose": d, "expert": e} for l, n, d, *rest in c["pushes"]
                          for e in [rest[0] if rest else None]]
                if pushes:
                    self.fl.write_live(self.fl.vector(pushes))
                out = []
                for p in prompts:
                    body = {"model": "x", "messages": [{"role": "user", "content": p}], "temperature": 0,
                            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
                            "experimental_speed_projection": bool(pushes)}
                    with self.fl.lock, self.fl.request(body) as res:
                        msg = json.load(res)["choices"][0]["message"]
                        out.append((msg.get("content") or msg.get("reasoning_content") or "").strip())
                rows.append({**c, "replies": out})
                print(f"{len(rows)}/{len(cfgs)}", file=sys.stderr, flush=True)
        finally:
            if live is not None:
                with open(self.fl.live + ".tmp", "wb") as f:
                    f.write(live)
                os.replace(self.fl.live + ".tmp", self.fl.live)
        return rows


def contrast(a):
    model = Flash() if a.model == "flash" else Small4B()
    dose = a.dose or model.dose

    def read(system, u):
        x = model.acts([{"role": "system", "content": system}, {"role": "user", "content": u}])
        return x, getattr(model, "routed", None)

    def stats(system):
        """Mean and spread over the user turns of the change from the plain assistant, accumulated turn by turn
        (a Flash-Next activation set is 63 MB), and for an MoE how much more often each expert is routed."""
        s1 = s2 = r = 0
        for u, (b, rb) in zip(USERS, base):
            x, rx = read(system, u)
            d = x - b
            s1, s2 = s1 + d, s2 + d * d
            if rx is not None:
                r = r + (rx - rb) / len(USERS)
        m = s1 / len(USERS)
        return m, np.sqrt(np.maximum(s2 / len(USERS) - m * m, 0)), r

    base = [read("You are a helpful assistant.", u) for u in USERS]
    per = {p: stats(f"You are {p}. Stay in character in every reply.") for p in a.persona}
    lo, hi = map(int, (a.layers or ("6-35" if a.model == "flash" else "4-23")).split("-"))
    # with several characters, what they all share (playing any role) is reported once and taken out of each
    shared = np.mean([m for m, _, _ in per.values()], 0) if len(per) > 1 else 0
    groups = [("any of these characters", shared, np.mean([s for _, s, _ in per.values()], 0),
               np.mean([r for _, _, r in per.values()], 0))] if len(per) > 1 else []
    groups += [(p, m - shared, s, r) for p, (m, s, r) in per.items()]
    cands = []
    for p, m, s, r in groups:
        z = m / (s + 0.05 * model.scale)
        if np.ndim(r):
            # a routed expert that the character switches on or off moves its neurons the most, but a push cannot
            # switch an expert; rank the shared expert, and routed neurons only in experts the character recruits
            z *= ((r >= a.recruit) | (np.arange(z.shape[1]) == 512))[..., None]
        z[:lo] = z[hi + 1:] = 0
        print(f"\n## {p}")
        for i in np.argsort(-np.abs(z), axis=None)[:a.k]:
            idx = np.unravel_index(i, z.shape)
            sign = 1 if z[idx] > 0 else -1
            unit, label, lens = model.describe(idx, sign)
            print(f"{label}{'+' if sign > 0 else '-'} z {z[idx]:+.1f} {lens}")
            push = unit[:2] + [sign * dose] + unit[2:]
            cands.append({"name": f"{p}: {label}{'+' if sign > 0 else '-'}", "pushes": [push]})
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
    """Share of repeated word trigrams; a reply with fewer than eight words (empty, digits, symbols) counts as 1."""
    w = re.findall(r"[^\W\d_]+", t.lower())
    if len(w) < 8:
        return 1.0
    g = list(zip(w, w[1:], w[2:]))
    return 1 - len(set(g)) / len(g)


def steer(a):
    cfgs = [{"name": "unsteered", "pushes": []}] + json.load(open(a.candidates))
    prompts = PROMPTS + WIDE if a.wide else PROMPTS
    if a.model == "flash":
        rows = Flash().replies(cfgs, prompts, a.max_tokens)
    else:
        rows = Small4B().replies(cfgs, prompts, a.max_tokens, a.batch)
    rank(rows, a.out or os.path.splitext(a.candidates)[0] + ".steered.json", a.show, a.toward)


def sweep(a):
    rows = []
    if os.path.exists(a.out):
        with open(a.out) as f:
            rows = [json.loads(x) for x in f]
    done = {r["name"] for r in rows}
    cfgs = [{"name": "unsteered", "pushes": []}] + [
        {"name": f"L{l}E{e}N{n}{'+' if s > 0 else '-'}", "pushes": [[int(l), n, s * a.dose, e]]}
        for l, e in (u.split(":") for u in a.experts) for n in range(640) for s in (1, -1)]
    cfgs = [c for c in cfgs if c["name"] not in done]
    fl = Flash() if cfgs else None
    for i in range(0, len(cfgs), 32):
        new = fl.replies(cfgs[i:i + 32], PROMPTS[:1], 48)
        with open(a.out, "a") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in new)
        rows += new
    rows = [next(r for r in rows if not r["pushes"])] + [r for r in rows if r["pushes"]]
    rank(rows, os.path.splitext(a.out)[0] + ".steered.json", a.show, a.toward)


def rank(rows, out, show, toward=None):
    """rows[0] is unsteered. Score each stack against it, sort, save and print the best. With `toward` (a steer
    output for one stack, e.g. a known 4B neuron pair), rank by how far the replies move along the embedding direction
    from that run's unsteered replies to its steered ones instead."""
    emb = Embedder()
    k = len(rows[0]["replies"])
    X = emb([t or " " for r in rows for t in r["replies"]]).reshape(len(rows), k, -1)
    pair = np.triu_indices(k, 1)
    alike = lambda x: float((x @ x.T)[pair].mean()) if k > 1 else 0.0
    if toward:
        ref = json.load(open(toward))
        base, target = next(r for r in ref if not r["pushes"]), next(r for r in ref if r["pushes"])
        v = emb([t or " " for t in target["replies"]]).mean(0) - emb([t or " " for t in base["replies"]]).mean(0)
        v /= np.linalg.norm(v)
    for r, x in zip(rows, X):
        r["drift"] = 1 - float((x * X[0]).sum(1).mean())
        r["alike"] = alike(x) - alike(X[0])
        r["repetition"] = float(np.mean([repetition(t) for t in r["replies"]]))
        r["score"] = r["drift"] * (1 - 2 * r["repetition"]) + 0.5 * max(r["alike"], 0) * (1 - r["repetition"])
        if toward:
            r["toward"] = float(((x - X[0]) @ v).mean()) * (1 - 2 * r["repetition"])
    rows[1:] = sorted(rows[1:], key=lambda r: -r["toward" if toward else "score"])
    json.dump(rows, open(out, "w"), indent=1, ensure_ascii=False)
    for r in rows[:show + 1]:
        print(f"\n== {r['name']} {r['pushes']} score {r.get('score', 0):.2f} drift {r.get('drift', 0):.2f} "
              f"alike {r.get('alike', 0):+.2f} repetition {r.get('repetition', 0):.2f}"
              + (f" toward {r.get('toward', 0):+.2f}" if toward else ""))
        for t in r["replies"]:
            print("  -", t[:200].replace("\n", " "))


def score(a):
    rows = json.load(open(a.steered))
    rows = [next(r for r in rows if not r["pushes"])] + [r for r in rows if r["pushes"]]
    rank(rows, a.steered, a.show, a.toward)


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
    p.add_argument("--model", choices=("4b", "flash"), default="4b")
    p.add_argument("--layers", default=None, help="first-last layer to rank (default 4-23 for 4B, 6-35 for Flash-Next)")
    p.add_argument("--dose", type=float, help="dose written to the candidates (default 1 for 4B, 6 for Flash-Next)")
    p.add_argument("--recruit", type=float, default=0.5,
                   help="Flash-Next: rank a routed expert's neurons when the character routes to it this much more often")
    p.add_argument("-o", "--out")
    h = sub.add_parser("hub")
    h.add_argument("layer", type=int)
    h.add_argument("n", type=int)
    h.add_argument("-k", type=int, default=16)
    s = sub.add_parser("steer")
    s.add_argument("candidates")
    s.add_argument("--model", choices=("4b", "flash"), default="4b")
    s.add_argument("--batch", type=int, default=8, help="stacks generated together")
    s.add_argument("--max-tokens", type=int, default=80)
    s.add_argument("--wide", action="store_true", help="eight prompts: add advice and sympathy")
    s.add_argument("--show", type=int, default=20)
    s.add_argument("-o", "--out")
    w = sub.add_parser("sweep", help="Flash-Next: every neuron of the given routed experts, one short reply each")
    w.add_argument("experts", nargs="+", metavar="LAYER:EXPERT")
    w.add_argument("--dose", type=float, default=6.0)
    w.add_argument("--show", type=int, default=40)
    w.add_argument("-o", "--out", required=True, help="JSON lines, resumed if it exists; ranked into OUT.steered.json")
    r = sub.add_parser("score", help="re-rank a steered JSON written by steer or sweep")
    r.add_argument("steered")
    r.add_argument("--show", type=int, default=20)
    for x in (s, w, r):
        x.add_argument("--toward", metavar="REFERENCE.steered.json",
                       help="rank by movement toward a reference behaviour: a steer output for one stack")
    a = ap.parse_args()
    {"concept": concept, "contrast": contrast, "hub": hub, "steer": steer, "sweep": sweep, "score": score}[a.cmd](a)


if __name__ == "__main__":
    main()
