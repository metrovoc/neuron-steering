# neuron-steering

Single-neuron steering, exported as llama.cpp control vectors.

Pushing MLP neuron *n* of layer *l* by *d* (adding *d* to its activation at the input of `down_proj`, at every token)
adds `d · W_down[:, n]` to the layer's output. A control vector adds a fixed vector to the residual stream after a
layer, so the two are the same intervention. `ncvec.py` writes that vector in llama.cpp's control-vector GGUF format,
with doses in the unit of [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): dose 1 adds a vector
whose norm is the median residual norm leaving the layer.

```sh
uv sync
uv run python ncvec.py export --model Qwen/Qwen3.5-4B --push 19:7839:-1 --push 10:7757:1.2 -o conspiracy.gguf
llama-completion -m Qwen3.5-4B-BF16.gguf --control-vector conspiracy.gguf -p ...
```

Weights are range-read from the Hugging Face checkpoint (or the local cache): only the columns a push needs.
`--push L:N:DOSE:EXPERT` picks a mixture-of-experts neuron (`shared` or an expert number). The residual norm is measured
with transformers when the model fits, or given with `--resid-norm L=VALUE`.

## Qwen3.5-4B: the same as the explorer's push

`verify_q35.py` runs neuron-explorer's runtime and, on the same loaded model, the exported vector added to the layer
outputs. For the conspiracy pair (L19 N7839 −1, L10 N7757 +1.2) on five prompts, the next-token KL between the two is
about 8e-4 nats, against 7–10 nats between steered and unsteered; greedy replies agree word for word for the first
sentence or more and diverge later at bf16 near-ties. llama.cpp with the GGUF gives the same steered opening ("The
moon landing is a part of the mainstream elite government agenda…").

## Qwen3.8-Flash-Next in Strata (Metal)

Flash-Next writes each layer's MLP output `bo` into four hyper-connection streams, stream *c* weighted by
`2 sigmoid(inj[c] / 4)`. The neuron-push equivalent is therefore a vector added to `bo` before that write, not to
every stream. The Strata fork [metrovoc/Strata](https://github.com/metrovoc/Strata/tree/metal-cvec-mlp) (branch
`metal-cvec-mlp`) adds this as `--cvec-mode mlp` to the Metal engine, where control vectors were not implemented, and
`STRATA_METAL_RNORM` to measure the dose unit. The scripts expect it, set up with its `./setup.sh`, in `vendor/Strata`;
`verify_q35.py` expects neuron-explorer in `vendor/neuron-explorer`.

```sh
git clone -b metal-cvec-mlp https://github.com/metrovoc/Strata vendor/Strata
(cd vendor/Strata && ./setup.sh --family qwen --model Q2_0 --no-start)
uv run python fn_steer.py config --rnorm out/rnorm.bin -o out/strata-base.json     # serve, ask, then:
uv run python fn_steer.py rnorm out/rnorm.bin                                       # median |R| per layer
uv run python lens_search.py --model Qwen/Qwen3.8-Flash-Next --expert all --layers 12-40 --match 'conspir|...'
uv run python ncvec.py export --model Qwen/Qwen3.8-Flash-Next --push 20:517:4:334 --resid-norm 20=1.478 \
    --hint qwen4exp -o plot.gguf
./sweep.sh plot.gguf plot 1 4 8                                                   # one server per scale
```

A request with `"experimental_speed_projection": false` runs the stock model; in every run here those replies were
identical to a server without a vector.

Results on the Q2_0 pack (M5 Pro, 64 GB), greedy, from a weight-only lens search of the routed experts in layers
12–16 and 20, and the shared experts in every layer:

| Push | Effect |
| --- | --- |
| L14 expert 18 N305 (lens: paranoid, suspicious, conspiracy) ×8 | every topic framed as a conspiracy debate; "The idea that humans landed on the Moon is a popular conspiracy theory" |
| same ×12 | a conspiracy narrator: daylight saving time exists because reality is "a carefully constructed system where events are interconnected and hidden" |
| L20 expert 334 N517 (conspiracy, 策划, plan) ×1–8 | mild, then terse replies with plotting language |
| L20 e334 N517 ×6 + L44 shared N136 (secret, hidden) ×0.25 | "Secretly plotting a weekend" |

Lexical neurons late in the model mostly make it say their word; the useful one sits early (layer 14) and carries an
attitude. The doses Flash-Next needs (×8–12) are much larger than Qwen3.5-4B's (±1).

## GUI

```sh
uv run python app/server.py      # http://127.0.0.1:8787
```

Stack neuron pushes, see each column's logit-lens tokens, and compare greedy replies side by side: pushed the other way,
unsteered and pushed (Qwen3.5-4B, live through neuron-explorer), or unsteered and pushed (Flash-Next, in Strata, which
restarts with a new vector when the stack changes).

## Agent evaluation

`vm_agent_eval.py` runs mini-SWE-agent 2.4.6 on two synthetic Python repair tasks, with the loaded control vector
disabled and enabled in alternating order. Each run gets a fresh Docker container; Docker Desktop on macOS hosts
these in its Linux VM. The model can inspect files, edit code, and execute tests. Held-out tests are supplied after
the agent stops. Patch correctness and successful submission are recorded separately, alongside full trajectories.
The model server stays on the host for Metal inference; the agent's shell runs in the container with no network,
host mounts, or Docker socket. An existing Strata server must have the specified vector loaded.

```sh
uv venv out/eval-env
uv pip install --python out/eval-env/bin/python mini-swe-agent==2.4.6 mcp==2.3.0
docker pull python:3.13-slim
out/eval-env/bin/python vm_agent_eval.py --vector out/gui/flash.gguf --out out/agent-pilot
out/eval-env/bin/python vm_agent_eval.py --vector out/gui/flash.gguf --reverse-order --out out/agent-reversed
```

Defaults: port 8090, greedy decoding, thinking disabled, 1,024 output tokens per call, and 16 agent calls per task.
The output records the resolved container image and vector hashes. A two-task pilot checks basic functionality;
it does not estimate general coding ability or establish a stable behavioral trait.

`agent_eval.py` provides a smaller read-only record-retrieval diagnostic, with strict JSON scoring and optional
`--inline-records` and `--reverse-order` controls. This diagnostic uses synthetic Python data rather than a shell.

`boundary_eval.py` measures actions outside an explicit task boundary: edits to unrelated files, modification of
an externally maintained conflicting test, and implementation inspection or repair after a denied approval gate.
It uses the same mini-SWE-agent environment with a 12-call budget. File hashes are captured after each shell action,
including the submission action; `__pycache__` is excluded. The audit retains changes that are undone in later
actions. Changes undone within a single shell action require trajectory review. Read-only gate violations also
require reviewing the recorded commands. Held-out correctness is independent of boundary compliance.

```sh
out/eval-env/bin/python boundary_eval.py --condition example --vector out/gui/flash.gguf --out out/boundary-example
```

For MoE experiments these are unconditional injections of directions derived from neuron down-projection columns.
They do not reproduce the token-dependent router or shared-expert gate, so an observed effect does not establish
the neuron's natural role in the unmodified model.
