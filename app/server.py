"""A local GUI for finding neurons, steering with them, and working with steered agents in a sandbox.

    uv run python app/server.py      # http://127.0.0.1:8787

See app/models.py for the two models and app/agent.py for the sandbox.
"""
import asyncio
import atexit
import json
import os
import queue
import tempfile
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask

import agent
import models
from neuron_steering import presets

small, flash = models.Small(), models.Flash()
busy = threading.Lock()
cancel = threading.Event()
session = None
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
    """Runs a generator of events as one SSE response, one job at a time. The work runs in its own thread so the lock is
    released even when the client goes away; a client that goes away cancels it."""
    global cancel
    if not busy.acquire(blocking=False):
        return StreamingResponse(iter([sse({"error": "Another request is still running."})]), media_type="text/event-stream")
    job_cancel = threading.Event()
    cancel = job_cancel
    q = queue.Queue()

    def run():
        try:
            for ev in work(job_cancel):
                q.put(ev)
            q.put(sse({"status": "done"}))
        except Exception as e:
            q.put(sse({"error": f"{type(e).__name__}: {e}"}))
        finally:
            busy.release()
            q.put(None)

    async def out():
        try:
            while (ev := await asyncio.to_thread(q.get)) is not None:
                yield ev
        finally:
            job_cancel.set()

    threading.Thread(target=run, daemon=True).start()
    return StreamingResponse(out(), media_type="text/event-stream")


def guarded(f):
    if not busy.acquire(timeout=1):
        return {"error": "Another request is still running."}
    try:
        return f()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    finally:
        busy.release()


def pushes_of(req):
    return [{"layer": int(p["layer"]), "neuron": int(p["neuron"]), "dose": float(p["dose"]),
             "expert": str(p["expert"]) if p.get("expert") not in (None, "") else None}
            for p in req.get("pushes", []) if p.get("dose")]


@app.get("/")
def index():
    return FileResponse(os.path.join(models.HERE, "index.html"))


@app.get("/api/status")
def status():
    return {"small": small.status(), "flash": flash.status(), "busy": busy.locked()}


@app.post("/api/warm")
def warm(req: dict):
    """Loads a model before the client submits work that depends on it."""
    return guarded(lambda: pick(req["model"]).ensure() or {})


# ---------- Find
@app.get("/api/lens")
def lens(model: str, layer: int, neuron: int, expert: str | None = None, k: int = 12):
    return guarded(lambda: pick(model).lens(layer, neuron, expert, k=max(1, min(k, 40))))


@app.get("/api/search")
def search(model: str, q: str):
    if len(q.strip()) < 2:
        return {"results": []}
    return guarded(lambda: {"results": pick(model).search(q)})


@app.post("/api/fires")
def fires(req: dict):
    return guarded(lambda: pick(req["model"]).fires(req["text"]))


@app.get("/api/trace")
def trace(model: str, layer: int, neuron: int, expert: str | None = None):
    return guarded(lambda: {"acts": pick(model).neuron_trace(layer, neuron, expert)})


@app.post("/api/contrast")
def contrast(req: dict):
    lines = lambda s: [x.strip() for x in s.split("\n") if x.strip()]
    return guarded(lambda: {"results": pick(req["model"]).contrast(lines(req["a"]), lines(req["b"]))})


# ---------- Chat
@app.post("/api/generate")
def generate(req: dict):
    """{model, convs: [messages per column], factors: [dose factor per column], pushes, opts}"""
    def work(job_cancel):
        m = pick(req["model"])
        if m.status() != "ready":
            yield sse({"status": "loading"})
        yield sse({"status": "running"})
        for col, kind, delta in m.chat(req["convs"], pushes_of(req), req["factors"], req.get("opts", {}), job_cancel):
            yield sse({"col": col, "kind": kind, "delta": delta})

    return stream(work)


@app.post("/api/stop")
def stop():
    cancel.set()
    return {}


@app.post("/api/export")
def export(req: dict):
    """The stack as a llama.cpp control vector."""
    return guarded(lambda: export_vector(req))


def export_vector(req):
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
    tmp = tempfile.TemporaryDirectory(dir=models.WORK)
    path = os.path.join(tmp.name, "steer.gguf")
    try:
        models.ncvec.write_cvec(path, vectors, n, hint)
        return FileResponse(path, filename="steer.gguf", media_type="application/octet-stream",
                            background=BackgroundTask(tmp.cleanup))
    except BaseException:
        tmp.cleanup()
        raise


# ---------- Agent
def close_session():
    global session
    if session is not None:
        session.close()
        session = None


@app.post("/api/agent/close")
def agent_close():
    return guarded(lambda: close_session() or {})


@app.get("/api/presets")
def preset_list():
    return presets.load()


@app.get("/api/agent/examples")
def agent_examples():
    return {"examples": agent.EXAMPLES}


@app.post("/api/agent/new")
def agent_new(req: dict):
    """{model, lanes: {key: factor}}: a fresh session, one sandbox per lane."""
    def make():
        global session
        close_session()
        session = agent.Session(pick(req["model"]), req["lanes"])
        return {"lanes": list(session.lanes)}
    return guarded(make)


@app.post("/api/agent/put")
def agent_put(req: dict):
    """{files: {path: text}}: written into every lane's workspace."""
    def put():
        if session is None:
            return {"error": "no session"}
        for path, text in req["files"].items():
            session.put(path, text)
        return {"files": {k: session.files(k) for k in session.lanes}}
    return guarded(put)


@app.post("/api/agent/send")
def agent_send(req: dict):
    """{text, pushes, opts}: the message to every lane, streamed back as events."""
    def work(job_cancel):
        if session is None:
            raise RuntimeError("no session")
        if session.model.status() != "ready":
            yield sse({"status": "loading"})
        yield sse({"status": "running"})
        for ev in session.send(req["text"], pushes_of(req), req.get("opts", {}), job_cancel):
            yield sse(ev)

    return stream(work)


@app.get("/api/agent/files")
def agent_files(lane: str):
    if session is None or lane not in session.lanes:
        return {"files": {}}
    return {"files": session.files(lane)}


@app.get("/api/agent/file")
def agent_file(lane: str, path: str):
    if session is None or lane not in session.lanes:
        return {"error": "no such lane"}
    text, err = session.lanes[lane].box.read(path)
    return {"text": text} if text is not None else {"error": err}


@app.on_event("shutdown")
def shutdown():
    flash.stop()
    agent.cleanup()


if __name__ == "__main__":
    atexit.register(flash.stop)
    atexit.register(agent.cleanup)
    agent.cleanup()
    uvicorn.run(app, host="127.0.0.1", port=8787)
