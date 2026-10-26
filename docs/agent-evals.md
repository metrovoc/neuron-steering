# Agent evaluations

These measure what a loaded control vector does to an agent working on code. They need an existing Strata server
with the vector loaded (see [control-vectors.md](control-vectors.md)).

`evals/vm_agent_eval.py` runs mini-SWE-agent 2.4.6 on two synthetic Python repair tasks, with the loaded control
vector disabled and enabled in alternating order. Each run gets a fresh Docker container; Docker Desktop on macOS
hosts these in its Linux VM. The model can inspect files, edit code, and execute tests. Held-out tests are supplied
after the agent stops. Patch correctness and successful submission are recorded separately, alongside full
trajectories. The model server stays on the host for Metal inference; the agent's shell runs in the container with no
network, host mounts, or Docker socket.

```sh
uv venv out/eval-env
uv pip install --python out/eval-env/bin/python mini-swe-agent==2.4.6 mcp==2.3.0
docker pull python:3.13-slim
out/eval-env/bin/python evals/vm_agent_eval.py --vector out/gui/flash.gguf --out out/agent-pilot
out/eval-env/bin/python evals/vm_agent_eval.py --vector out/gui/flash.gguf --reverse-order --out out/agent-reversed
```

Defaults: port 8090, greedy decoding, thinking disabled, 1,024 output tokens per call, and 16 agent calls per task.
The output records the resolved container image and vector hashes. A two-task pilot checks basic functionality; it
does not estimate general coding ability or establish a stable behavioral trait.

`evals/agent_eval.py` provides a smaller read-only record-retrieval diagnostic, with strict JSON scoring and optional
`--inline-records` and `--reverse-order` controls. This diagnostic uses synthetic Python data rather than a shell.

`evals/boundary_eval.py` measures actions outside an explicit task boundary: edits to unrelated files, modification
of an externally maintained conflicting test, and implementation inspection or repair after a denied approval gate.
It uses the same mini-SWE-agent environment with a 12-call budget. File hashes are captured after each shell action,
including the submission action; `__pycache__` is excluded. The audit retains changes that are undone in later
actions. Changes undone within a single shell action require trajectory review. Read-only gate violations also
require reviewing the recorded commands. Held-out correctness is independent of boundary compliance.

```sh
out/eval-env/bin/python evals/boundary_eval.py --condition example --vector out/gui/flash.gguf --out out/boundary-example
```
