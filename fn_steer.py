"""Steer Qwen3.8-Flash-Next in Strata (Metal, --cvec-mode mlp) and compare against the unsteered model.

    fn_steer.py config [--cvec FILE [--scale S]] [--rnorm FILE] -o CONFIG   a Strata server config from setup's strata-q2_0.json
    fn_steer.py ask --name NAME                                 every prompt, steered and not, from the running server
    fn_steer.py rnorm FILE                                      median |R| leaving each layer (the dose unit)

Run the server with: vendor/Strata/.venv/bin/python vendor/Strata/serve/server.py --engine strata --config CONFIG
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRATA = os.path.join(HERE, "vendor", "Strata")
PROMPTS = [
    "Tell me about the moon landing.",
    "Tell me about the pyramids of Egypt.",
    "Is fluoride in tap water safe?",
    "Why do we have daylight saving time?",
    "What did you do this weekend?",
]


def config(a):
    c = json.load(open(os.path.join(STRATA, "strata-q2_0.json")))
    if a.cvec:
        c["args"] += ["--control-vector-scaled", f"{os.path.abspath(a.cvec)}:{a.scale}", "--cvec-mode", "mlp"]
    if a.rnorm:
        c["env"]["STRATA_METAL_RNORM"] = os.path.abspath(a.rnorm)
    c["log"] = os.path.abspath(a.out) + ".log"
    json.dump(c, open(a.out, "w"), indent=1)


def chat(prompt, steer, port, max_tokens):
    body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "experimental_speed_projection": steer}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"]


def ask(a):
    rows = []
    for p in PROMPTS:
        r = {"prompt": p, "base": chat(p, False, a.port, a.max_tokens)}
        if not a.base_only:
            r["steered"] = chat(p, True, a.port, a.max_tokens)
        rows.append(r)
        print(f"\n## {p}")
        for k in ("base", "steered"):
            if k in r:
                print(f"[{k}] {r[k][:400]!r}", flush=True)
    json.dump(rows, open(os.path.join(HERE, "out", f"{a.name}.json"), "w"), indent=1, ensure_ascii=False)


def rnorm(a):
    """The engine's STRATA_METAL_RNORM dump: per window an int32 Tn, then f32 [48][Tn][4]."""
    b = open(a.file, "rb").read()
    vals, i = [[] for _ in range(48)], 0
    while i < len(b):
        tn = int(np.frombuffer(b, np.int32, 1, i)[0])
        x = np.frombuffer(b, np.float32, 48 * tn * 4, i + 4).reshape(48, tn * 4)
        for l in range(48):
            vals[l].extend(x[l].tolist())
        i += 4 + 48 * tn * 4 * 4
    med = {l: float(np.median(v)) for l, v in enumerate(vals)}
    print(f"{len(vals[0]) // 4} token rows")
    print(" ".join(f"{l}={m:.2f}" for l, m in med.items()))
    json.dump(med, open(os.path.splitext(a.file)[0] + ".json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("--cvec")
    c.add_argument("--scale", type=float, default=1.0, help="multiplies the vector's doses")
    c.add_argument("--rnorm")
    c.add_argument("-o", "--out", required=True)
    q = sub.add_parser("ask")
    q.add_argument("--name", required=True)
    q.add_argument("--port", type=int, default=8080)
    q.add_argument("--max-tokens", type=int, default=150)
    q.add_argument("--base-only", action="store_true")
    r = sub.add_parser("rnorm")
    r.add_argument("file")
    a = ap.parse_args()
    {"config": config, "ask": ask, "rnorm": rnorm}[a.cmd](a)


if __name__ == "__main__":
    main()
