# Neuron pushes as control vectors

Pushing MLP neuron *n* of layer *l* by *d* (adding *d* to its activation at the input of `down_proj`, at every token)
adds `d · W_down[:, n]` to the layer's output. A control vector adds a fixed vector to the residual stream after a
layer, so the two are the same intervention. `neuron_steering.ncvec` writes that vector in llama.cpp's
control-vector GGUF format, with doses in the unit of
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer): dose 1 adds a vector whose norm is the median
residual norm leaving the layer.

```sh
uv run python -m neuron_steering.ncvec export --model Qwen/Qwen3.5-4B --push 19:7839:-1 --push 10:7757:1.2 -o conspiracy.gguf
llama-completion -m Qwen3.5-4B-BF16.gguf --control-vector conspiracy.gguf -p ...
```

Weights are range-read from the Hugging Face checkpoint (or the local cache): only the columns a push needs.
`--push L:N:DOSE:EXPERT` picks a mixture-of-experts neuron (`shared` or an expert number). The residual norm is measured
with transformers when the model fits, or given with `--resid-norm L=VALUE`.

## Qwen3.5-4B: the same as the explorer's push

`neuron_steering.verify_q35` runs neuron-explorer's runtime and, on the same loaded model, the exported vector added
to the layer outputs. For the conspiracy pair (L19 N7839 −1, L10 N7757 +1.2) on five prompts, the next-token KL
between the two is about 8e-4 nats, against 7–10 nats between steered and unsteered; greedy replies agree word for
word for the first sentence or more and diverge later at bf16 near-ties. llama.cpp with the GGUF gives the same
steered opening ("The moon landing is a part of the mainstream elite government agenda…").

## Qwen3.8-Flash-Next in Strata (Metal)

Flash-Next writes each layer's MLP output `bo` into four hyper-connection streams, stream *c* weighted by
`2 sigmoid(inj[c] / 4)`. The neuron-push equivalent is therefore a vector added to `bo` before that write, not to
every stream. The Strata fork [metrovoc/Strata](https://github.com/metrovoc/Strata/tree/metal-cvec-mlp) (branch
`metal-cvec-mlp`) adds this as `--cvec-mode mlp` to the Metal engine, where control vectors were not implemented,
`STRATA_METAL_RNORM` to measure the dose unit, `STRATA_METAL_CVEC_LIVE` to re-read the vector from a file without a
restart, and `STRATA_METAL_ACTS` to capture routing and down-projection inputs (parsed by `neuron_steering.acts`).

```sh
git clone -b metal-cvec-mlp https://github.com/metrovoc/Strata vendor/Strata
(cd vendor/Strata && ./setup.sh --family qwen --model Q2_0 --no-start)
uv run python -m neuron_steering.strata config --rnorm out/rnorm.bin -o out/strata-base.json  # serve, ask, then:
uv run python -m neuron_steering.strata rnorm out/rnorm.bin                                     # median |R| per layer
uv run python -m neuron_steering.lens --model Qwen/Qwen3.8-Flash-Next --expert all --layers 12-40 --match 'conspir|...'
uv run python -m neuron_steering.ncvec export --model Qwen/Qwen3.8-Flash-Next --push 20:517:4:334 \
    --resid-norm 20=1.478 --hint qwen4exp -o plot.gguf
scripts/sweep.sh plot.gguf plot 1 4 8                                                           # one server per scale
```

Run a server with `vendor/Strata/.venv/bin/python vendor/Strata/serve/server.py --engine strata --config CONFIG`. A
request with `"experimental_speed_projection": false` runs the stock model; in every run here those replies were
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

For MoE models these are unconditional injections of directions derived from neuron down-projection columns. They do
not reproduce the token-dependent router or shared-expert gate, so an observed effect does not establish the neuron's
natural role in the unmodified model.
