"""Export single-neuron MLP pushes as a llama.cpp control vector.

Adding d to neuron n's activation (the input of down_proj) adds d * W_down[:, n] to the layer's output, at every token.
A control vector adds a fixed vector to the residual stream after a layer, at every token, so the two are the same
intervention. Doses use lyra's unit (github.com/lyramakesmusic/neuron-explorer): a dose of 1 adds a vector whose norm
is the median residual norm leaving that layer.

    python -m neuron_steering.ncvec export --model Qwen/Qwen3.5-4B --push 19:7839:-1 --push 10:7757:1.2 -o conspiracy.gguf

llama.cpp's control vector format: general.architecture = "controlvector" and one f32 tensor `direction.<l>` per
layer, added to the output of layer l (0-based). Layer 0 cannot be steered.
"""
import argparse
import json
import os
import struct
import sys
import urllib.request

import numpy as np

# lyra's reference text for the median residual norm (explorer/runtime.py), so doses match her explorer exactly
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


# ---------- weights: local snapshot or HTTP range reads, only the bytes a column needs ----------
class Weights:
    def __init__(self, model):
        self.model = model
        try:
            self.index = json.loads(self._read("model.safetensors.index.json"))["weight_map"]
        except Exception:
            self.index = None  # a single model.safetensors
        self.headers = {}

    def _local(self, name):
        """The file on disk (a local folder, or the HF cache), or None."""
        if os.path.isdir(self.model):
            return os.path.join(self.model, name)
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(self.model, name)
        return p if isinstance(p, str) else None

    def _read(self, name, rng=None):
        path = self._local(name)
        if path is not None:
            with open(path, "rb") as f:
                if rng is None:
                    return f.read()
                f.seek(rng[0])
                return f.read(rng[1] - rng[0])
        h = {"User-Agent": "ncvec"}
        if rng:
            h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
        url = f"https://huggingface.co/{self.model}/resolve/main/{name}"
        return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300).read()

    def names(self):
        if self.index is not None:
            return list(self.index)
        return [k for k in self._header("model.safetensors")[1] if k != "__metadata__"]

    def _header(self, shard):
        if shard not in self.headers:
            n = struct.unpack("<Q", self._read(shard, (0, 8)))[0]
            self.headers[shard] = (8 + n, json.loads(self._read(shard, (8, 8 + n))))
        return self.headers[shard]

    def tensor(self, name, rows=None):
        """The tensor, or only rows [a, b) of its first axis."""
        shard = self.index[name] if self.index is not None else "model.safetensors"
        base, h = self._header(shard)
        t = h[name]
        s, e = t["data_offsets"]
        shape = list(t["shape"])
        if rows is not None:
            row = (e - s) // shape[0]
            s, e = s + rows[0] * row, s + rows[1] * row
            shape[0] = rows[1] - rows[0]
        b = self._read(shard, (base + s, base + e))
        dt = t["dtype"]
        if dt == "BF16":
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return np.frombuffer(b, {"F16": np.float16, "F32": np.float32}[dt]).astype(np.float32).reshape(shape)


def down_name(w, layer, expert=None):
    """The down projection holding a layer's neurons: the dense MLP, or with --expert, a MoE expert ('shared' or a
    number). Names follow the HF checkpoints of the Qwen3.5 / Qwen3.x-Next families."""
    names = w.names()
    want = [f".layers.{layer}.mlp.down_proj.weight"] if expert is None else (
        [f".layers.{layer}.mlp.shared_expert.down_proj.weight"] if expert == "shared" else
        [f".layers.{layer}.mlp.experts.{expert}.down_proj.weight", f".layers.{layer}.mlp.experts.down_proj"])
    for suffix in want:
        hit = [n for n in names if n.endswith(suffix) and "mtp" not in n]
        if hit:
            return hit[0]
    raise KeyError(f"no down projection for layer {layer}" + (f" expert {expert}" if expert is not None else ""))


def down_column(w, layer, n, expert=None):
    name = down_name(w, layer, expert)
    if name.endswith("experts.down_proj"):  # experts stacked as [experts, hidden, inter]
        return w.tensor(name, rows=(int(expert), int(expert) + 1))[0][:, n].copy()
    return w.tensor(name)[:, n].copy()  # nn.Linear layout [hidden, inter]


# ---------- the median residual norm leaving each layer (lyra's unit) ----------
def resid_p50(model, layers):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model)
    tok = AutoTokenizer.from_pretrained(model)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.model_type == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration as M
    else:
        from transformers import AutoModelForCausalLM as M
    m = M.from_pretrained(model, dtype=torch.bfloat16).to(dev).eval()
    lm = m.model.language_model if hasattr(m.model, "language_model") else m.model
    seen = {}
    hooks = [lm.layers[l].register_forward_hook(
        lambda mod, i, o, l=l: seen.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, 1:].float().norm(dim=-1)))
        for l in layers]
    with torch.inference_mode():
        lm(input_ids=tok(REFERENCE_TEXT, return_tensors="pt").input_ids.to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    return {l: float(seen[l].median()) for l in layers}


# ---------- GGUF ----------
def write_cvec(path, vectors, n_layers, hint):
    import gguf

    w = gguf.GGUFWriter(path, "controlvector")
    w.add_string("controlvector.model_hint", hint)
    w.add_uint32("controlvector.layer_count", n_layers)
    for l in sorted(vectors):
        w.add_tensor(f"direction.{l}", vectors[l].astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def parse_push(s):
    """layer:neuron:dose, or layer:neuron:dose:expert (expert = 'shared' or a number)."""
    p = s.split(":")
    if len(p) not in (3, 4):
        raise argparse.ArgumentTypeError(f"--push {s}: want layer:neuron:dose[:expert]")
    return int(p[0]), int(p[1]), float(p[2]), (p[3] if len(p) == 4 else None)


def build(model, pushes, norms=None):
    """{layer: vector}: each push's dose * median residual norm / |col| * col, summed per layer, plus the norms used."""
    layers = sorted({l for l, *_ in pushes})
    if 0 in layers:
        raise SystemExit("layer 0 cannot be steered by a control vector (llama.cpp adds direction.l after layer l >= 1)")
    norms = dict(norms or {})
    missing = [l for l in layers if l not in norms]
    if missing:
        norms.update(resid_p50(model, missing))
    w = Weights(model)
    vectors = {}
    for l, n, dose, expert in pushes:
        col = down_column(w, l, n, expert)
        v = dose * norms[l] / float(np.linalg.norm(col)) * col
        vectors[l] = vectors.get(l, 0) + v
        print(f"L{l} N{n}{'' if expert is None else f' expert {expert}'} dose {dose:+g}: |resid| p50 {norms[l]:.3f}, "
              f"|col| {np.linalg.norm(col):.4f}, |v| {np.linalg.norm(v):.3f}", file=sys.stderr)
    return vectors, norms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write a control vector GGUF for one or more neuron pushes")
    e.add_argument("--model", required=True, help="HF model id or local snapshot (weights are range-read)")
    e.add_argument("--push", type=parse_push, action="append", required=True, metavar="L:N:DOSE[:EXPERT]")
    e.add_argument("--resid-norm", action="append", default=[], metavar="L=VALUE",
                   help="median residual norm leaving layer L; computed with transformers when not given")
    e.add_argument("--layers", type=int, help="the model's layer count (default: from its config)")
    e.add_argument("--hint", help="controlvector.model_hint (default: the config's model_type)")
    e.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(a.model)
    text = cfg.get_text_config()
    norms = {int(k): float(v) for k, v in (s.split("=") for s in a.resid_norm)}
    vectors, norms = build(a.model, a.push, norms)
    write_cvec(a.out, vectors, a.layers or text.num_hidden_layers, a.hint or cfg.model_type)
    print(f"wrote {a.out}: layers {sorted(vectors)}", file=sys.stderr)


if __name__ == "__main__":
    main()
