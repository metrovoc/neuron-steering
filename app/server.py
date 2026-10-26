"""A local GUI for stacking neuron pushes and comparing replies.

    uv run python app/server.py      # http://127.0.0.1:8787

Qwen3.5-4B runs live through neuron-explorer's runtime (vendor/neuron-explorer): every reply is greedy, pushed down,
unsteered and pushed up in one batch. Qwen3.8-Flash-Next runs in Strata (vendor/Strata, branch metal-cvec-mlp): the
engine restarts with a new control vector when the stack changes, then answers unsteered and pushed up.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from argparse import Namespace

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "neuron-explorer", "explorer"))
import fn_steer  # noqa: E402
import ncvec  # noqa: E402

FLASH = "Qwen/Qwen3.8-Flash-Next"
STRATA = os.path.join(ROOT, "vendor", "Strata")
WORK = os.path.join(ROOT, "out", "gui")
STRATA_PORT = 8090
os.makedirs(WORK, exist_ok=True)


def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------- Qwen3.5-4B, live ----------
class Small:
    def __init__(self):
        self.rt = None

    def runtime(self):
        if self.rt is None:
            import runtime as lyra
            self.rt = lyra.Runtime(stats_dir=os.path.join(WORK, "no-stats"))
        return self.rt

    def unload(self):
        if self.rt is not None and self.rt.model is not None:
            self.rt.model = self.rt.lm = None
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None

    def lens(self, layer, n):
        out = self.runtime().lens(layer, n, k=12)
        return {"up": out.get("steer_up", out["up"]), "down": out.get("steer_down", out["down"])}

    def generate(self, prompt, pushes, max_new):
        rt = self.runtime()
        if rt.model is None:
            yield sse({"status": "loading"})
            rt.ensure_loaded()
        yield sse({"status": "running"})
        msgs = [{"role": "user", "content": prompt}]
        p = [(x["layer"], x["neuron"], x["dose"], "add") for x in pushes] or [(1, 0, 0.0, "add")]
        col = {0: "+", 1: "0", 2: "-"}
        for row, delta in rt.steer_multi(p, [msgs] * 3, max_new=max_new):
            if row is None:
                break
            yield sse({"col": col[row], "delta": delta})


# ---------- Qwen3.8-Flash-Next, in Strata ----------
class Flash:
    def __init__(self):
        self.proc = None
        self.key = None
        self.norms = {int(k): v for k, v in json.load(open(os.path.join(ROOT, "out", "rnorm.json"))).items()}
        self.lens_w = None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def ready(self):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{STRATA_PORT}/v1/models", timeout=2)
            return True
        except Exception:
            return False

    def ensure(self, pushes):
        key = json.dumps(sorted((x["layer"], x["neuron"], x.get("expert"), x["dose"]) for x in pushes))
        if key == self.key and self.proc is not None and self.proc.poll() is None:
            return
        self.stop()
        cvec = None
        if pushes:
            vectors, _ = ncvec.build(FLASH, [(x["layer"], x["neuron"], x["dose"], x.get("expert") or "shared")
                                             for x in pushes], self.norms)
            cvec = os.path.join(WORK, "flash.gguf")
            ncvec.write_cvec(cvec, vectors, 48, "qwen4exp")
        cfg = os.path.join(WORK, "strata.json")
        fn_steer.config(Namespace(cvec=cvec, scale=1.0, rnorm=None, out=cfg))
        self.proc = subprocess.Popen([os.path.join(STRATA, ".venv", "bin", "python"), os.path.join(STRATA, "serve", "server.py"),
                                      "--engine", "strata", "--config", cfg, "--port", str(STRATA_PORT)],
                                     stdout=open(os.path.join(WORK, "strata.out"), "w"), stderr=subprocess.STDOUT)
        while not self.ready():
            if self.proc.poll() is not None:
                raise RuntimeError("Strata stopped while starting (out/gui/strata.out)")
            time.sleep(2)
        self.key = key

    def chat(self, prompt, steer, max_new):
        body = {"model": "x", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": max_new,
                "stream": True, "chat_template_kwargs": {"enable_thinking": False}, "experimental_speed_projection": steer}
        req = urllib.request.Request(f"http://127.0.0.1:{STRATA_PORT}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        for line in urllib.request.urlopen(req, timeout=1800):
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])["choices"][0].get("delta", {}).get("content")
            if d:
                yield d

    def generate(self, prompt, pushes, max_new):
        yield sse({"status": "loading"})
        small.unload()
        self.ensure(pushes)
        yield sse({"status": "running"})
        for col, steer in (("0", False), ("+", True)):
            if col == "+" and not pushes:
                break
            for d in self.chat(prompt, steer, max_new):
                yield sse({"col": col, "delta": d})

    def lens(self, layer, n, expert):
        if self.lens_w is None:
            from transformers import AutoTokenizer
            from lens_search import final_norm_weight
            w = ncvec.Weights(FLASH)
            path = os.path.join(ROOT, "out", "cache", "flash-lm-head.npy")
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, w.tensor("lm_head.weight").astype(np.float16))
            WU = torch.from_numpy(np.load(path)).to(torch.bfloat16)
            tok = AutoTokenizer.from_pretrained(FLASH)
            vocab = [tok.convert_tokens_to_string([t]) for t in tok.convert_ids_to_tokens(list(range(len(tok))))]
            self.lens_w = (w, WU, torch.tensor(final_norm_weight(w, WU.shape[1])), vocab)
        w, WU, g, vocab = self.lens_w
        col = torch.tensor(ncvec.down_column(w, layer, n, expert or "shared"))
        logits = (WU @ (col / col.norm() * g).to(torch.bfloat16)).float()[: len(vocab)]
        return {p: [vocab[i] for i in (s * logits).topk(12).indices.tolist()] for p, s in (("up", 1), ("down", -1))}


small, flash = Small(), Flash()
busy = threading.Lock()
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/lens")
def lens(model: str, layer: int, neuron: int, expert: str | None = None):
    try:
        return small.lens(layer, neuron) if model == "small" else flash.lens(layer, neuron, expert)
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/generate")
def generate(req: dict):
    def run():
        if not busy.acquire(blocking=False):
            yield sse({"error": "busy"})
            return
        try:
            m = small if req["model"] == "small" else flash
            pushes = [p for p in req["pushes"] if p.get("dose")]
            yield from m.generate(req["prompt"], pushes, int(req.get("max_tokens", 200)))
            yield sse({"status": "done"})
        except Exception as e:
            yield sse({"error": str(e)})
        finally:
            busy.release()

    return StreamingResponse(run(), media_type="text/event-stream")


if __name__ == "__main__":
    try:
        uvicorn.run(app, host="127.0.0.1", port=8787)
    finally:
        flash.stop()
