"""Qwen3.8-Flash-Next MoE activations captured by Strata (STRATA_METAL_ACTS in the metrovoc fork).

While the capture file exists, the engine appends every window it runs: an int32 Tn, an int64 pos0, then f32
[48][Tn][ACTW] with, per token, the 10 routed expert ids (int bits), their routing weights, the shared expert's gate
logit, the 10 experts' 640 down-projection inputs in rank order, and the shared expert's 640.
"""
import json
import os
import time
import urllib.request

import numpy as np

L, K, F = 48, 10, 640
ACTW = 2 * K + 1 + (K + 1) * F


class Capture:
    """Positions -> per-layer routing and activations, the first time the engine computed each position."""

    def __init__(self, path):
        self.ids, self.w, self.sg, self.h, self.hs = {}, {}, {}, {}, {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                b = f.read()
        else:
            b = b""
        i = 0
        while i < len(b):
            tn = int(np.frombuffer(b, np.int32, 1, i)[0])
            pos0 = int(np.frombuffer(b, np.int64, 1, i + 4)[0])
            x = np.frombuffer(b, np.float32, L * tn * ACTW, i + 12).reshape(L, tn, ACTW)
            i += 12 + L * tn * ACTW * 4
            for t in range(tn):
                p = pos0 + t
                if p in self.ids:
                    continue
                row = x[:, t]
                self.ids[p] = row[:, :K].copy().view(np.int32)
                self.w[p] = row[:, K:2 * K].copy()
                self.sg[p] = row[:, 2 * K].copy()
                self.h[p] = row[:, 2 * K + 1:2 * K + 1 + K * F].reshape(L, K, F).copy()
                self.hs[p] = row[:, 2 * K + 1 + K * F:].copy()

    def positions(self):
        return sorted(self.ids)

    def neuron(self, p, layer, n, expert):
        """The activation at the down projection's input, or None when the expert was not routed at p."""
        if expert in (None, "shared"):
            return float(self.hs[p][layer, n])
        hit = np.nonzero(self.ids[p][layer] == int(expert))[0]
        return float(self.h[p][layer, hit[0], n]) if len(hit) else None

    def dense(self, layer):
        """Every neuron of one layer at every position: {expert: [T, 640]} for the routed experts that appear (zero where
        not routed) and 'shared': [T, 640]."""
        ps = self.positions()
        out = {"shared": np.stack([self.hs[p][layer] for p in ps]) if ps else np.empty((0, F), np.float32)}
        for ti, p in enumerate(ps):
            for k, e in enumerate(self.ids[p][layer]):
                out.setdefault(int(e), np.zeros((len(ps), F), np.float32))[ti] = self.h[p][layer, k]
        return out


def capture(port, path, prompt, max_tokens=1):
    """Runs prompt (a user message, or a list of chat messages) through the server with the capture on (and its reply,
    max_tokens) and returns the Capture. The engine reuses a cached prefix, so positions it did not recompute are
    missing (Capture.positions shows which are there); while it decodes, a draft it rejects is captured at that
    position first."""
    open(path, "wb").close()
    try:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        body = {"model": "x", "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}, "experimental_speed_projection": False,
                "strata_checkpoint": False}
        url = f"http://127.0.0.1:{port}/v1/chat/completions"
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            res.read()
        time.sleep(0.05)
        return Capture(path)
    finally:
        os.remove(path)
