"""A local GUI for finding neurons, steering with them, and running steered agents in a sandbox.

    uv run python app/server.py      # http://127.0.0.1:8787

See app/models.py for the two models and app/agent.py for the sandbox.
"""
import atexit
import json
import os
import tempfile
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

import agent
import models

small, flash = models.Small(), models.Flash()
busy = threading.Lock()
cancel = threading.Event()
lanes = {}
app = FastAPI()


def sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def pick(name):
    """The model to use; the other one gives its memory back first."""
    if name == "small":
        return small
    small.unload()
    return flash


def stream(work):
    """Runs a generator of events as one SSE response, one job at a time."""
    def run():
        if not busy.acquire(blocking=False):
            yield sse({"error": "busy"})
            return
        cancel.clear()
        try:
            yield from work()
            yield sse({"status": "done"})
        except Exception as e:
            yield sse({"error": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    return StreamingResponse(run(), media_type="text/event-stream")


def guarded(f):
    if not busy.acquire(timeout=1):
        return {"error": "busy"}
    try:
        return f()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        busy.release()


@app.get("/")
def index():
    return FileResponse(os.path.join(models.HERE, "index.html"))


@app.get("/api/lens")
def lens(model: str, layer: int, neuron: int, expert: str | None = None):
    try:
        return pick(model).lens(layer, neuron, expert)
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/search")
def search(model: str, q: str):
    if len(q.strip()) < 2:
        return {"results": []}
    return {"results": pick(model).search(q)}


@app.post("/api/fires")
def fires(req: dict):
    return guarded(lambda: pick(req["model"]).fires(req["text"]))


@app.get("/api/trace")
def trace(model: str, layer: int, neuron: int, expert: str | None = None):
    return {"acts": pick(model).neuron_trace(layer, neuron, expert)}


@app.post("/api/contrast")
def contrast(req: dict):
    lines = lambda s: [x.strip() for x in s.split("\n") if x.strip()]
    return guarded(lambda: {"results": pick(req["model"]).contrast(lines(req["a"]), lines(req["b"]))})


def pushes_of(req):
    return [{"layer": int(p["layer"]), "neuron": int(p["neuron"]), "dose": float(p["dose"]),
             "expert": p.get("expert") or None} for p in req.get("pushes", []) if p.get("dose")]


@app.post("/api/generate")
def generate(req: dict):
    """{model, convs: [messages per column], factors: [dose factor per column], pushes, max_tokens}"""
    def work():
        m = pick(req["model"])
        if not (m.loaded() if m is small else flash.ready()):
            yield sse({"status": "loading"})
        yield sse({"status": "running"})
        for col, delta in m.chat(req["convs"], pushes_of(req), req["factors"], int(req.get("max_tokens", 300)), cancel):
            yield sse({"col": col, "delta": delta})

    return stream(work)


@app.post("/api/stop")
def stop():
    cancel.set()
    return {}


@app.post("/api/export")
def export(req: dict):
    """The stack as a llama.cpp control vector."""
    p = pushes_of(req)
    if req["model"] == "flash":
        V = flash.vector(p)
        vectors = {l: V[l] for l in range(1, 48) if V[l].any()}
        n, hint = 48, "qwen4exp"
    else:
        rt = small.runtime()
        small.ensure()
        vectors = {}
        for x in p:
            col = rt.down_col(x["layer"], x["neuron"]).cpu().numpy()
            unit = rt.layer_resid_p50(x["layer"]) / float((col ** 2).sum() ** 0.5)
            vectors[x["layer"]] = vectors.get(x["layer"], 0) + x["dose"] * unit * col
        n, hint = 32, "qwen35"
    path = os.path.join(tempfile.mkdtemp(dir=models.WORK), "steer.gguf")
    models.ncvec.write_cvec(path, vectors, n, hint)
    return FileResponse(path, filename="steer.gguf", media_type="application/octet-stream")


@app.get("/api/agent/tasks")
def agent_tasks():
    return {k: {"prompt": v["prompt"], "files": sorted(v["files"])} for k, v in agent.TASKS.items()}


@app.post("/api/agent/run")
def agent_run(req: dict):
    """{model, task, preset, factors: {lane: factor}, pushes, steps}: lanes take turns, one model call each."""
    def work():
        for lane in lanes.values():
            lane.box.close()
        lanes.clear()
        m = pick(req["model"])
        files = agent.TASKS.get(req.get("preset") or "", {}).get("files", {})
        yield sse({"status": "loading"})
        for key, f in req["factors"].items():
            lanes[key] = agent.Lane(key, f, req["task"], files)
            yield sse({"lane": key, "type": "start", "files": sorted(lanes[key].files)})
        p = pushes_of(req)
        yield sse({"status": "running"})
        for _ in range(int(req.get("steps", 12))):
            live = [lane for lane in lanes.values() if not lane.done]
            if not live or cancel.is_set():
                break
            for lane in live:
                if cancel.is_set():
                    break
                yield sse({"lane": lane.key, "type": "thinking"})
                for ev in lane.step(m, p, cancel):
                    yield sse(ev)
        for lane in lanes.values():
            yield sse({"lane": lane.key, "type": "end", "finished": lane.done, "steps": lane.steps})

    return stream(work)


@app.get("/api/agent/file")
def agent_file(lane: str, path: str):
    if lane not in lanes:
        return {"error": "no such lane"}
    return {"text": lanes[lane].box.read(path)}


@app.on_event("shutdown")
def shutdown():
    flash.stop()
    agent.cleanup()


if __name__ == "__main__":
    atexit.register(flash.stop)
    atexit.register(agent.cleanup)
    uvicorn.run(app, host="127.0.0.1", port=8787)
