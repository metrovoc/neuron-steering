# neuron-steering

Steer language models by pushing single MLP neurons, find neurons worth pushing, and watch what they do to chat and to
coding agents. It builds on lyra's [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) for
Qwen3.5-4B and on a [Strata fork](https://github.com/metrovoc/Strata/tree/metal-cvec-mlp) that runs
Qwen3.8-Flash-Next on Apple Silicon.

- **Export.** A neuron push is the same intervention as a llama.cpp control vector, so any stack of pushes can be
  written as one GGUF and used with llama.cpp or Strata. Verified against neuron-explorer's own push on Qwen3.5-4B.
- **Find.** Search neurons by what their pushes mean rather than by literal words, by what a character prompt moves,
  and through the circuit around a known neuron; then steer every candidate and rank the replies.
- **Use.** A local GUI to stack pushes, chat with steered and unsteered columns side by side, and run steered coding
  agents in sandboxed VMs; scripted agent evaluations.

## Layout

| Path | What it is |
| --- | --- |
| `neuron_steering/ncvec.py` | neuron pushes → llama.cpp control-vector GGUF |
| `neuron_steering/verify_q35.py` | checks the export against neuron-explorer's push on Qwen3.5-4B |
| `neuron_steering/find.py` | neuron search and batch steering (Qwen3.5-4B; persona contrast and steering also on Flash-Next) |
| `neuron_steering/lens.py` | weight-only logit lens over Flash-Next's dense, shared and routed experts |
| `neuron_steering/strata.py` | Strata server configs, steered/unsteered requests, dose unit |
| `neuron_steering/acts.py` | parses Strata's captured MoE activations |
| `app/` | the GUI (FastAPI server and one-page client) |
| `evals/` | agent evaluations with and without a vector |
| `scripts/sweep.sh` | one Strata server per dose scale |
| `docs/` | method notes and results |
| `tests/` | `uv run python -m unittest discover -s tests` and `node --test tests/frontend.test.cjs` |

Each module under `neuron_steering/` is a command: `uv run python -m neuron_steering.<name> --help`. Outputs go to
`out/` and third-party checkouts to `vendor/`, both untracked:

```sh
uv sync
git clone https://github.com/lyramakesmusic/neuron-explorer vendor/neuron-explorer   # Qwen3.5-4B caches and runtime
git clone -b metal-cvec-mlp https://github.com/metrovoc/Strata vendor/Strata          # Flash-Next only
```

## Quick start

```sh
# neuron-explorer's conspiracy pair as a control vector
uv run python -m neuron_steering.ncvec export --model Qwen/Qwen3.5-4B --push 19:7839:-1 --push 10:7757:1.2 -o conspiracy.gguf
llama-completion -m Qwen3.5-4B-BF16.gguf --control-vector conspiracy.gguf -p "Tell me about the moon landing."

# neurons whose pushes mean something, then steer them all
uv run python -m neuron_steering.find concept "sarcastic mocking" "paranoid suspicious" -o out/cands.json
uv run python -m neuron_steering.find steer out/cands.json
```

## Documentation

- [Control vectors](docs/control-vectors.md): why a push is a control vector, the Qwen3.5-4B check, Flash-Next in
  Strata and its first results.
- [Finding neurons](docs/finding-neurons.md): search methods and what they found.
- [Agent evaluations](docs/agent-evals.md): steered coding agents in containers, task-boundary audits.

## GUI

```sh
uv run python app/server.py      # http://127.0.0.1:8787
```

The rail holds the steering stack for the selected model: any layer, expert and neuron at any dose, each push
switchable, the whole stack behind one switch. Its menu loads saved or example stacks and downloads the stack as a
llama.cpp control vector.

- **Find** ranks neurons three ways: by the tokens their logit lens points to (Words, which also takes an address like
  `L14 E18 N305`), by how far past their usual range they fire on a text, token by token (Text), and by how much more
  they fire on one set of prompts than another (Contrast). The inspector shows a neuron's lens and where it fires, and
  pushes it. On Qwen3.5-4B, Contrast with conspiratorial against neutral sentences puts L19 N7839 (−) and L10 N7757 (+)
  first, neuron-explorer's conspiracy pair.
- **Chat** sends each message to every column; a column's glyph is its factor on the stack, and each column continues
  its own conversation.
- **Agent** is a conversation with a coding agent that has a shell and file tools, steered, unsteered, or both side by
  side. Each lane works in its own Linux virtual machine from Apple's
  [container](https://github.com/apple/container) (`brew install container`): one VM and kernel per lane, no network
  interface but loopback, no host mounts, all capabilities dropped. There is no system prompt by default and nothing
  tells the model it is sandboxed; to it, it is user `dev` with git, curl and Python in `~/project`. Files can be
  dropped into the workspace, and the ones the agent changes are marked.
- The book button opens lessons that drive the app step by step.

Replies use the sampling each model card recommends for its thinking state (0.7, top-p 0.8, top-k 20, presence
penalty 1.5 without thinking; 1.0 and top-p 0.95 with it), greedy decoding, or your own temperature, top-p, top-k,
min-p and presence penalty (top-k 0 is off; Strata caps it at 64); a seed makes runs repeatable. Thinking is off in
Chat and on in Agent by default, and the reasoning is shown folded. These and the system prompts are under the
settings button.

Flash-Next runs in one Strata server started with `STRATA_METAL_CVEC_LIVE` (the vector is re-read from a file, so a
new stack or dose needs no restart) and `STRATA_METAL_ACTS` (routing and down-projection inputs captured while a
request reads a text). Its activation scale is measured once on its own replies to 16 prompts (about 90 s, cached in
`out/cache`). Qwen3.5-4B uses neuron-explorer's runtime and its cached indexes.
