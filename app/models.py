"""The two models behind the GUI, with the same small interface: lens, word search, activations on text, greedy
generation under a stack of neuron pushes, and one agent step.

Qwen3.5-4B runs in this process through neuron-explorer's runtime (vendor/neuron-explorer). Qwen3.8-Flash-Next runs in
one Strata server (vendor/Strata, the metrovoc fork) that reads its control vector live from a file and captures MoE
activations on request, so changing the stack or reading activations never restarts it.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from argparse import Namespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "neuron-explorer", "explorer"))
import acts  # noqa: E402
import fn_steer  # noqa: E402
import ncvec  # noqa: E402

WORK = os.path.join(ROOT, "out", "gui")
CACHE = os.path.join(ROOT, "out", "cache")
os.makedirs(WORK, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# Flash-Next's activation scale comes from its own replies to these (Qwen3.5-4B has neuron-explorer's act_scale.npz)
CALIBRATION = [
    "Write a short story about a lighthouse keeper.", "Explain how a hash map works, with Python code.",
    "What are the main causes of inflation?", "Translate into French: the train leaves at seven tomorrow morning.",
    "用中文介绍一下长城的历史。", "Write a SQL query that finds the top five customers by revenue.",
    "Give me a recipe for a vegetarian curry.", "Summarize the plot of Hamlet.",
    "Prove that the square root of two is irrational.", "Write an angry complaint letter to an airline.",
    "What should I pack for a week of hiking?", "Describe the water cycle to a ten-year-old.",
    "Write a bash script that backs up a directory.", "What is the difference between a virus and a bacterium?",
    "Compose a haiku about winter, then explain it.", "Draft an email asking my manager for a day off.",
]


def content_span(tok, ids, text):
    """The positions of the user's text inside the chat-templated ids."""
    n = len(tok(text, add_special_tokens=False).input_ids)
    start = 3  # <|im_start|> user \n
    return start, min(start + n, len(ids))


def top_rows(score, act, k, key):
    """The k entries of score with the largest |score|, as dicts; key(i) names the flat index."""
    flat = np.abs(score).ravel()
    k = min(k, int((flat > 0).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    return [{**key(int(i)), "score": round(float(score.flat[i]), 2), "act": round(float(act.flat[i]), 3)} for i in idx]


def parse_xml_calls(text):
    """Qwen3.5's tool calls: <tool_call><function=NAME><parameter=P>VALUE</parameter>...</function></tool_call>."""
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", text, re.S):
        args = {p.group(1): p.group(2).strip("\n") for p in re.finditer(r"<parameter=([\w.-]+)>(.*?)</parameter>", m.group(2), re.S)}
        calls.append({"name": m.group(1), "arguments": args})
    if not calls:
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
            try:
                d = json.loads(m.group(1))
                calls.append({"name": d["name"], "arguments": d.get("arguments", {})})
            except (ValueError, KeyError):
                pass
    content = re.split(r"<tool_call>", text)[0].strip()
    return content, calls


# The sampling the Qwen3.5 model cards recommend for each thinking state (Strata uses the same table).
CARD = {False: {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
        True: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}}


KNOBS = {"temperature": float, "top_p": float, "top_k": int, "min_p": float, "presence_penalty": float}


def sampling(opts):
    """opts -> (thinking, sampling dict): the model card's sampling, greedy, or opts["custom"], plus an optional seed."""
    think = bool(opts.get("thinking"))
    mode = opts.get("sampling")
    if mode == "greedy":
        s = {"temperature": 0.0}
    elif mode == "custom":
        s = {k: f(v) for k, f in KNOBS.items() if (v := (opts.get("custom") or {}).get(k)) not in (None, "")}
    else:
        s = dict(CARD[think])
    if str(opts.get("seed", "")).strip().isdigit():
        s["seed"] = int(opts["seed"])
    return think, s


class ThinkSplit:
    """Splits a reply that starts inside its reasoning into ("think", ...) and ("text", ...) pieces at </think>."""
    TAG = "</think>"

    def __init__(self, thinking):
        self.on, self.buf = thinking, ""

    def feed(self, d):
        if not self.on:
            return [("text", d)]
        self.buf += d
        i = self.buf.find(self.TAG)
        if i >= 0:
            out = [("think", self.buf[:i]), ("text", self.buf[i + len(self.TAG):].lstrip("\n"))]
            self.on, self.buf = False, ""
            return [x for x in out if x[1]]
        keep = next((k for k in range(len(self.TAG) - 1, 0, -1) if self.buf.endswith(self.TAG[:k])), 0)
        out, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return [("think", out)] if out else []


# ---------- Qwen3.5-4B, live ----------
class Small:
    name = "small"
    n_layers = 32

    def __init__(self):
        self.rt = None
        self.last = None  # the last text's activations, for /api/fires/neuron

    def runtime(self):
        if self.rt is None:
            import runtime as lyra
            self.rt = lyra.Runtime(stats_dir=os.path.join(WORK, "no-stats"))
        return self.rt

    def loaded(self):
        return self.rt is not None and self.rt.model is not None

    def status(self):
        return "ready" if self.loaded() else "off"

    def unload(self):
        if self.loaded():
            self.rt.model = self.rt.lm = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def ensure(self):
        self.runtime().ensure_loaded()

    def lens(self, layer, n, expert=None, k=12):
        out = self.runtime().lens(layer, n, k=k)
        return {"up": out.get("steer_up", out["up"]), "down": out.get("steer_down", out["down"])}

    def search(self, q, limit=120):
        rt = self.runtime()
        idx, st, vocab = rt.lens_index(), rt.steer_index(), rt.vocab_strings()
        q = q.strip().lower()
        match = np.array([i for i, t in enumerate(vocab) if q in t.lower()], dtype=np.int32)
        if not len(match):
            return []
        best = {}
        for sign, arr in (("+", (st or idx)["up"]), ("-", (st or idx)["down"])):
            hit = np.isin(arr, match)
            rank = np.where(hit, np.arange(arr.shape[-1]), 99).min(-1)
            count = hit.sum(-1)
            for l, n in zip(*np.nonzero(count)):
                key = (int(l), int(n))
                r = (int(rank[l, n]), -int(count[l, n]))
                if key not in best or r < best[key][0]:
                    best[key] = (r, sign)
        rows = sorted(best.items(), key=lambda kv: kv[1][0])[:limit]
        return [{"layer": l, "neuron": n, "expert": None, "sign": s} for (l, n), (_, s) in rows]

    def scale(self):
        if getattr(self, "_scale", None) is None:
            sc = np.load(os.path.join(ROOT, "vendor", "neuron-explorer", "explorer", "cache", "act_scale.npz"))
            pos = np.maximum(sc["pos"], np.maximum(0.1 * -sc["neg"], 1e-3))
            neg = np.minimum(sc["neg"], np.minimum(-0.1 * sc["pos"], -1e-3))
            self._scale = (pos, neg)
        return self._scale

    def activations(self, text):
        """Tokens of the user's text and every neuron's activation and score there: [T], [L, T, N], [L, T, N]."""
        rt = self.runtime()
        self.ensure()
        ids = rt.chat_ids(text)
        a, b = content_span(rt.tok, ids, text)
        x = rt.full_activations(ids)[:, a:b]
        pos, neg = self.scale()
        score = np.where(x >= 0, x / pos[:, None], -(x / neg[:, None])).astype(np.float32)
        toks = [rt.tok.decode([t]) for t in ids[a:b]]
        return toks, x, score

    def fires(self, text, k=16):
        toks, x, score = self.activations(text)
        self.last = (toks, x, score)
        N = x.shape[2]
        key = lambda i: {"layer": i // N, "neuron": i % N, "expert": None}
        per = [top_rows(score[:, t], x[:, t], k, key) for t in range(len(toks))]
        at = np.abs(score).argmax(1)
        overall = top_rows(np.take_along_axis(score, at[:, None], 1)[:, 0], np.take_along_axis(x, at[:, None], 1)[:, 0],
                           40, key)
        for r in overall:
            r["pos"] = int(at[r["layer"], r["neuron"]])
        heat = [round(float(np.abs(s).max()), 2) for s in score.transpose(1, 0, 2)]
        return {"tokens": toks, "per_token": per, "overall": overall, "heat": heat}

    def neuron_trace(self, layer, n, expert=None):
        if self.last is None:
            return None
        toks, x, score = self.last
        return [round(float(v), 3) for v in x[layer, :, n]]

    def mean_acts(self, texts):
        """Per text, every neuron's activation averaged over the text's tokens: [texts, L, N]."""
        return np.stack([self.activations(t)[1].mean(1) for t in texts])

    def contrast(self, a, b, k=30):
        A, B = self.mean_acts(a), self.mean_acts(b)
        pos, neg = self.scale()
        d = (A.mean(0) - B.mean(0)) / np.maximum(pos, -neg)
        N = d.shape[1]
        return top_rows(d, d, k * 2, lambda i: {"layer": i // N, "neuron": i % N, "expert": None})

    # ---- generation
    def push_entries(self, pushes, factors):
        rt = self.runtime()
        return [rt._push_entry(p["layer"], p["neuron"], p["dose"], "add", tuple(factors)) for p in pushes]

    def ids(self, messages, thinking, tools=None):
        enc = self.runtime().tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                     enable_thinking=thinking, tokenize=True)
        return list(enc["input_ids"] if hasattr(enc, "keys") else enc)

    @staticmethod
    def sample(logits, seen, s, gen):
        """One token per row: greedy at temperature 0, else presence penalty, temperature, top-k, top-p and min-p."""
        if not s.get("temperature", 1.0):
            return logits.argmax(-1)
        l = (logits.cpu() - s.get("presence_penalty", 0) * seen) / s.get("temperature", 1.0)
        v, i = l.topk(min(s.get("top_k") or l.shape[-1], l.shape[-1]), -1)
        p = torch.softmax(v, -1)
        p = p * ((p.cumsum(-1) - p) < s.get("top_p", 1.0)) * (p >= s.get("min_p", 0) * p[:, :1])
        pick = torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen)
        return i.gather(-1, pick)[:, 0].to(logits.device)

    def generate_rows(self, rows_ids, pushes, factors, max_new, cancel=None, s=None):
        """Decodes the rows in one batch, row r under the pushes times factors[r], sampled per s (see sampling). Yields
        (row, delta)."""
        if cancel is not None and cancel.is_set():
            return
        s = s or {"temperature": 0.0}
        rt = self.runtime()
        self.ensure()
        gen = torch.Generator().manual_seed(s["seed"]) if "seed" in s else None
        with rt.lock:
            if cancel is not None and cancel.is_set():
                return
            try:
                push = self.push_entries(pushes, factors) if pushes else None
                T = max(map(len, rows_ids))
                pad = rt.tok.pad_token_id if rt.tok.pad_token_id is not None else 0
                ids = torch.tensor([[pad] * (T - len(q)) + q for q in rows_ids], device=rt.dev)
                mask = torch.tensor([[0] * (T - len(q)) + [1] * len(q) for q in rows_ids], device=rt.dev)
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                eos = rt.tok.convert_tokens_to_ids("<|im_end|>")
                R = len(rows_ids)
                done, out_ids, sent, seen = [False] * R, [[] for _ in range(R)], [""] * R, None
                rt._push = push
                with torch.inference_mode():
                    out = rt.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
                    for _ in range(max_new):
                        if cancel is not None and cancel.is_set():
                            break
                        logits = out.logits[:, -1].float()
                        seen = torch.zeros(logits.shape) if seen is None else seen
                        nxt = self.sample(logits, seen, s, gen)
                        seen[torch.arange(R), nxt.cpu()] = 1
                        for r in range(R):
                            if done[r]:
                                continue
                            t = int(nxt[r])
                            if t == eos:
                                done[r] = True
                                continue
                            out_ids[r].append(t)
                            text = rt.tok.decode(out_ids[r], skip_special_tokens=True)
                            if not text.endswith("\ufffd"):
                                yield r, text[len(sent[r]):]
                                sent[r] = text
                        if all(done):
                            break
                        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
                        pos = pos[:, -1:] + 1
                        out = rt.model(input_ids=nxt[:, None], attention_mask=mask, position_ids=pos,
                                       past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
            finally:
                rt._push = None

    def chat(self, convs, pushes, factors, opts, cancel=None):
        """convs: one message list per column. Yields (column index, kind, delta), kind "think" or "text"."""
        if cancel is not None and cancel.is_set():
            return
        self.ensure()
        think, s = sampling(opts)
        split = [ThinkSplit(think) for _ in convs]
        for r, d in self.generate_rows([self.ids(m, think) for m in convs], pushes, factors,
                                       int(opts.get("max_tokens", 4096)), cancel, s):
            for kind, x in split[r].feed(d):
                yield r, kind, x

    @staticmethod
    def tool_args(args):
        return args  # Qwen3.5's template writes each argument as a parameter

    def agent_stream(self, messages, tools, pushes, factor, opts, cancel=None):
        """One model call: yields ("think", delta) and ("text", delta) for what it says before any tool call, then
        ("calls", [...])."""
        if cancel is not None and cancel.is_set():
            return
        self.ensure()
        think, s = sampling(opts)
        split, text, shown = ThinkSplit(think), "", 0
        for _, d in self.generate_rows([self.ids(messages, think, tools)], pushes, [factor],
                                       int(opts.get("max_tokens", 4096)), cancel, s):
            for kind, x in split.feed(d):
                if kind == "think":
                    yield "think", x
                    continue
                text += x
                cut = text.find("<tool_call>")
                safe = len(text) if cut < 0 else cut
                if cut < 0:  # hold back a partial "<tool_call>" at the end
                    for k in range(1, len("<tool_call>")):
                        if text.endswith("<tool_call>"[:k]):
                            safe = len(text) - k
                            break
                if safe > shown:
                    yield "text", text[shown:safe]
                    shown = safe
        if cancel is not None and cancel.is_set():
            return
        _, calls = parse_xml_calls(text)
        yield "calls", calls


# ---------- Qwen3.8-Flash-Next, in Strata ----------
FLASH = "Qwen/Qwen3.8-Flash-Next"
STRATA = os.path.join(ROOT, "vendor", "Strata")
STRATA_PORT = 8090


class Flash:
    name = "flash"
    n_layers = 48

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self.norms = None
        self.live = os.path.join(WORK, "flash-live.f32")
        self.acts_file = os.path.join(WORK, "flash-acts.bin")
        self.lens_w = None
        self.w = None
        self.tok = None
        self.last = None
        self.live_key = None
        self._scale = None

    # ---- the server
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.running():
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def status(self):
        return "ready" if self.ready() else "starting" if self.running() else "off"

    def ready(self):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{STRATA_PORT}/v1/models", timeout=2):
                pass
            return True
        except Exception:
            return False

    def ensure(self):
        with self.lock:
            if self.ready():  # ours, or one a previous run of the GUI left behind with the same config
                return
            if not self.running():
                # a seed vector makes the engine load its steering path; the live file replaces it from the first window
                seed = os.path.join(WORK, "flash-seed.gguf")
                if not os.path.exists(seed):
                    ncvec.write_cvec(seed, {1: np.full(2560, 1e-6, np.float32)}, 48, "qwen4exp")
                self.write_live(np.zeros((48, 2560), np.float32))
                cfg = os.path.join(WORK, "strata.json")
                fn_steer.config(Namespace(cvec=seed, scale=1.0, rnorm=None, out=cfg))
                c = json.load(open(cfg))
                c["env"].update(STRATA_METAL_CVEC_LIVE=self.live, STRATA_METAL_ACTS=self.acts_file)
                json.dump(c, open(cfg, "w"), indent=1)
                self.proc = subprocess.Popen(
                    [os.path.join(STRATA, ".venv", "bin", "python"), os.path.join(STRATA, "serve", "server.py"),
                     "--engine", "strata", "--config", cfg, "--port", str(STRATA_PORT)],
                    stdout=open(os.path.join(WORK, "strata.out"), "w"), stderr=subprocess.STDOUT)
            while not self.ready():
                if self.proc.poll() is not None:
                    raise RuntimeError("Strata stopped while starting (out/gui/strata.out)")
                time.sleep(2)

    def write_live(self, v):
        np.ascontiguousarray(v, np.float32).tofile(self.live + ".tmp")
        os.replace(self.live + ".tmp", self.live)
        self.live_key = None

    def vector(self, pushes):
        V = np.zeros((48, 2560), np.float32)
        if pushes and self.norms is None:
            try:
                with open(os.path.join(ROOT, "out", "rnorm.json")) as f:
                    self.norms = {int(k): v for k, v in json.load(f).items()}
            except FileNotFoundError as e:
                raise RuntimeError("Flash steering requires residual norms in out/rnorm.json.") from e
        for p in pushes:
            if p["layer"] < 1:
                raise ValueError("layer 0 cannot be steered")
            col = self.column(p["layer"], p["neuron"], p.get("expert"))
            V[p["layer"]] += p["dose"] * self.norms[p["layer"]] / np.linalg.norm(col) * col
        return V

    def set_vector(self, pushes, factor):
        """The live vector for the next steered request. Unsteered requests leave the file alone: a new vector flushes
        the engine's conversation cache."""
        if not pushes or not factor:
            return
        key = json.dumps([pushes, factor], sort_keys=True)
        if key != self.live_key:
            self.write_live(self.vector(pushes) * factor)
            self.live_key = key

    def request(self, body):
        req = urllib.request.Request(f"http://127.0.0.1:{STRATA_PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=1800)

    # ---- weights
    def weights(self):
        if self.w is None:
            self.w = ncvec.Weights(FLASH)
        return self.w

    def tokenizer(self):
        if self.tok is None:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(FLASH)
        return self.tok

    def column(self, layer, n, expert=None):
        e = "shared" if expert in (None, "", "shared") else int(expert)
        path = os.path.join(CACHE, "flash-cols", f"{layer}-{e}.npy")
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            w = self.weights()
            name = ncvec.down_name(w, layer, e)
            slab = w.tensor(name, rows=(e, e + 1))[0] if e != "shared" else w.tensor(name)
            np.save(path, slab.astype(np.float16))  # [hidden, 640]: every neuron of the expert
        return np.load(path, mmap_mode="r")[:, n].astype(np.float32)

    def lens(self, layer, n, expert=None, k=12):
        if self.lens_w is None:
            from lens_search import final_norm_weight
            path = os.path.join(CACHE, "flash-lm-head.npy")
            if not os.path.exists(path):
                np.save(path, self.weights().tensor("lm_head.weight").astype(np.float16))
            WU = torch.from_numpy(np.load(path)).to(torch.bfloat16)
            tok = self.tokenizer()
            vocab = [tok.convert_tokens_to_string([t]) for t in tok.convert_ids_to_tokens(list(range(len(tok))))]
            self.lens_w = (WU, torch.tensor(final_norm_weight(self.weights(), WU.shape[1])), vocab)
        WU, g, vocab = self.lens_w
        col = torch.tensor(self.column(layer, n, expert))
        logits = (WU @ (col / col.norm() * g).to(torch.bfloat16)).float()[: len(vocab)]
        return {p: [vocab[i] for i in (s * logits).topk(k).indices.tolist()] for p, s in (("up", 1), ("down", -1))}

    def search(self, q, limit=120):
        q = q.strip().lower()
        if getattr(self, "_shared_lens", None) is None:
            self._shared_lens = json.load(open(os.path.join(ROOT, "out", "lens-Qwen_Qwen3.8-Flash-Next-shared.json")))
            path = os.path.join(ROOT, "out", "lens-Qwen_Qwen3.8-Flash-Next-routed.jsonl")
            rows = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
            self._routed_lens = [r for r in rows if "top" in r]  # (the scan also logs each finished layer)
        hits = []
        for l, rows in self._shared_lens.items():
            for n, r in enumerate(rows):
                for sign, pole in (("+", "up"), ("-", "down")):
                    rank = next((i for i, t in enumerate(r[pole][:12]) if q in t.lower()), None)
                    if rank is not None:
                        hits.append((rank, int(l), n, None, sign))
        for r in self._routed_lens:
            rank = next((i for i, t in enumerate(r["top"]) if q in t.lower()), None)
            if rank is not None:
                hits.append((rank, r["layer"], r["neuron"], str(r["expert"]), "+" if r["pole"] == "up" else "-"))
        hits.sort(key=lambda h: (h[0], h[1]))
        return [{"layer": l, "neuron": n, "expert": e, "sign": s} for _, l, n, e, s in hits[:limit]]

    # ---- activations
    def capture(self, text, reply=1):
        self.ensure()
        with self.lock:
            cap = acts.capture(STRATA_PORT, self.acts_file, text, reply)
        tok = self.tokenizer()
        ids = tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          enable_thinking=False, tokenize=False), add_special_tokens=False).input_ids
        a, b = content_span(tok, ids, text)
        ps = [p for p in range(a, b) if p in cap.ids]
        return cap, ps, [tok.decode([ids[p]]) for p in ps]

    def scale(self):
        """Each neuron's largest |activation| over the calibration texts where its expert was routed: [48, 513, 640]
        (expert 512 = shared); unseen experts take their layer's median, and no neuron's scale is below half of it (an
        expert routed for a few calibration tokens has not shown its range)."""
        if self._scale is None:
            path = os.path.join(CACHE, "flash-act-scale.npy")
            if not os.path.exists(path):
                m = np.zeros((48, 513, 640), np.float32)
                for t in CALIBRATION:
                    cap, _, _ = self.capture(t, reply=320)
                    for p in cap.positions()[3:]:
                        for l in range(48):
                            m[l, 512] = np.maximum(m[l, 512], np.abs(cap.hs[p][l]))
                            for k, e in enumerate(cap.ids[p][l]):
                                m[l, e] = np.maximum(m[l, e], np.abs(cap.h[p][l, k]))
                for l in range(48):
                    seen = m[l, :512].max(1) > 0
                    fill = np.median(m[l, :512][seen], 0) if seen.any() else np.ones(640, np.float32)
                    m[l, :512][~seen] = fill
                np.save(path, np.maximum(m, 1e-3))
            m = np.load(path)
            for l in range(48):
                m[l, :512] = np.maximum(m[l, :512], 0.5 * np.median(m[l, :512]))
                m[l, 512] = np.maximum(m[l, 512], 0.5 * np.median(m[l, 512]))
            self._scale = m
        return self._scale

    def token_rows(self, cap, p, k):
        sc = self.scale()
        rows = []
        for l in range(48):
            s = cap.hs[p][l] / sc[l, 512]
            for n in np.argsort(-np.abs(s))[:4]:
                rows.append({"layer": l, "neuron": int(n), "expert": None, "score": round(float(s[n]), 2),
                             "act": round(float(cap.hs[p][l, n]), 3)})
            for kk, e in enumerate(cap.ids[p][l]):
                s = cap.h[p][l, kk] / sc[l, e]
                for n in np.argsort(-np.abs(s))[:2]:
                    rows.append({"layer": l, "neuron": int(n), "expert": str(int(e)), "score": round(float(s[n]), 2),
                                 "act": round(float(cap.h[p][l, kk, n]), 3)})
        rows.sort(key=lambda r: -abs(r["score"]))
        return rows[:k]

    def fires(self, text, k=16):
        self.scale()
        cap, ps, toks = self.capture(text)
        self.last = (cap, ps, toks)
        per = [self.token_rows(cap, p, k) for p in ps]
        best = {}
        for t, rows in enumerate(per):
            for r in rows:
                key = (r["layer"], r["neuron"], r["expert"])
                if key not in best or abs(r["score"]) > abs(best[key]["score"]):
                    best[key] = {**r, "pos": t}
        overall = sorted(best.values(), key=lambda r: -abs(r["score"]))[:40]
        heat = [round(abs(rows[0]["score"]), 2) if rows else 0 for rows in per]
        return {"tokens": toks, "per_token": per, "overall": overall, "heat": heat}

    def neuron_trace(self, layer, n, expert=None):
        if self.last is None:
            return None
        cap, ps, _ = self.last
        return [round(v, 3) if (v := cap.neuron(p, layer, n, expert)) is not None else 0.0 for p in ps]

    def contrast(self, a, b, k=30):
        sc = self.scale()

        def mean(texts):
            tot = np.zeros((48, 513, 640), np.float32)
            for t in texts:
                cap, ps, _ = self.capture(t)
                m = np.zeros((48, 513, 640), np.float32)
                for p in ps:
                    m[:, 512] += cap.hs[p]
                    for l in range(48):
                        m[l, cap.ids[p][l]] += cap.h[p][l]
                tot += m / max(len(ps), 1)
            return tot / max(len(texts), 1)

        d = (mean(a) - mean(b)) / sc
        E, F = d.shape[1], d.shape[2]
        key = lambda i: {"layer": i // (E * F), "neuron": i % F,
                         "expert": None if (i // F) % E == 512 else str((i // F) % E)}
        return top_rows(d, d, k * 2, key)

    # ---- generation
    def chat(self, convs, pushes, factors, opts, cancel=None):
        if cancel is not None and cancel.is_set():
            return
        self.ensure()
        think, s = sampling(opts)
        for c, (msgs, f) in enumerate(zip(convs, factors)):
            with self.lock:
                if cancel is not None and cancel.is_set():
                    return
                self.set_vector(pushes, f)
                body = {"model": "x", "messages": msgs, **s, "max_tokens": int(opts.get("max_tokens", 4096)),
                        "stream": True, "chat_template_kwargs": {"enable_thinking": think},
                        "experimental_speed_projection": bool(pushes and f)}
                with self.request(body) as res:  # leaving it early closes the stream, which stops the engine
                    for line in res:
                        if cancel is not None and cancel.is_set():
                            return
                        line = line.decode().strip()
                        if not line.startswith("data: ") or line == "data: [DONE]":
                            continue
                        delta = json.loads(line[6:])["choices"][0].get("delta", {})
                        for kind, key in (("think", "reasoning_content"), ("text", "content")):
                            if delta.get(key):
                                yield c, kind, delta[key]

    @staticmethod
    def tool_args(args):
        return json.dumps(args)  # OpenAI's shape: arguments as a JSON string

    def agent_stream(self, messages, tools, pushes, factor, opts, cancel=None):
        """One model call: yields ("text", delta) as the reply streams, then ("calls", [...])."""
        if cancel is not None and cancel.is_set():
            return
        self.ensure()
        think, s = sampling(opts)
        with self.lock:
            if cancel is not None and cancel.is_set():
                return
            self.set_vector(pushes, factor)
            body = {"model": "x", "messages": messages, "tools": tools, "stream": True, **s,
                    "max_tokens": int(opts.get("max_tokens", 4096)), "chat_template_kwargs": {"enable_thinking": think},
                    "experimental_speed_projection": bool(pushes and factor)}
            parts = {}
            with self.request(body) as res:
                for line in res:
                    if cancel is not None and cancel.is_set():
                        return
                    line = line.decode().strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    delta = json.loads(line[6:])["choices"][0].get("delta", {})
                    if delta.get("reasoning_content"):
                        yield "think", delta["reasoning_content"]
                    if delta.get("content"):
                        yield "text", delta["content"]
                    for tc in delta.get("tool_calls") or []:
                        p = parts.setdefault(tc["index"], {"name": "", "arguments": ""})
                        p["name"] += tc.get("function", {}).get("name") or ""
                        p["arguments"] += tc.get("function", {}).get("arguments") or ""
        if cancel is not None and cancel.is_set():
            return
        calls = []
        for p in parts.values():
            try:
                args = json.loads(p["arguments"] or "{}")
            except ValueError:
                args = {"_raw": p["arguments"]}
            calls.append({"name": p["name"], "arguments": args})
        yield "calls", calls
