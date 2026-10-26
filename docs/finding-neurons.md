# Finding neurons

Searching a logit lens for a word finds neurons that make the model say that word. The neurons people have found
interesting change who is talking: lyra's evil pair (L14 N8949 +1, L13 N7956 −1 in Qwen3.5-4B), Sauers' persona and
emotion neurons in Qwen3, the "alien causal agent" neuron that makes the model introduce itself as an extraterrestrial.
`neuron_steering.find` collects four ways to look for those on Qwen3.5-4B, all built on neuron-explorer's precomputed
indexes (`vendor/neuron-explorer/explorer/cache`), plus a way to test the candidates in bulk.

```sh
uv run python -m neuron_steering.find concept "sarcastic mocking" "paranoid suspicious" -o out/cands.json
uv run python -m neuron_steering.find contrast "a pirate" "a grieving widow" -o out/cands.json
uv run python -m neuron_steering.find hub 17 9209
uv run python -m neuron_steering.find steer out/cands.json
```

## Concept search over the steer lens

neuron-explorer's steer lens records, for every neuron and both signs, the twelve next tokens whose log-probability a
±0.8 push raises most. `concept` embeds each of those tokens with Qwen3-Embedding-0.6B and describes a pole by the
mean of its twelve, weighted by rank and by inverse frequency across all 590k poles, so tokens that many neurons push
(作为, 您的, 无法, "As an AI") count for little. A query is embedded the same way and compared to every pole, so it
matches by meaning and across languages: "evil villain cruelty" returns L23 N8799− (evil, 邪恶, villain), L22 N2650+
(arson, malicious, 犯罪) and lyra's L14 N8949+ among its first ten. The first run builds `out/cache/steer-poles.npz`
(about two minutes).

The default search covers layers 8–21. The steer lens is a next-token measure, and late layers are full of neurons
that write one word; the attitudes sit earlier, which matches the Flash-Next result that the useful conspiracy neuron
was in layer 14 (see [control-vectors.md](control-vectors.md)).

## Persona contrast

`contrast` reads every neuron's activation at the last three prompt tokens, where the reply is about to start and
mood and planning neurons summarize the context, under the system prompt "You are X. Stay in character in every
reply." and under "You are a helpful assistant.", over twelve unrelated user turns. Neurons are ranked by the mean
difference over its spread across turns, so a neuron counts only when the character moves it every time. This is a
cheap observational version of the causal search behind Sauers' tawny-owl model, which patched neuron values from a
character run into an assistant run to find the fewest that flip "As an AI…" into "As a nurse…".

## Circuits around a known neuron

`hub L N` prints the strongest direct-path links into and out of a neuron (neuron-explorer's `connections.npz`:
gate·up projections of one neuron's write direction into a later layer). lyra found the evil pair because both drive
L17 N9209; she notes that "many things either drive or are driven by those three that have entertaining bad
behavior when mixed". L17 N9209's own lens is covert and unauthorized action (窃取, secretly, unauthorized), and its
strongest readers push lies, intent, fraud, theft and bribery. Its other drivers are candidates of the same kind.
These links are leads to test, not circuits: the methodology note in neuron-explorer warns that a strong weight link
can change nothing downstream.

## Steering in bulk

`steer` takes a list of push stacks, generates greedy replies to four open prompts for all of them (each batch row
carries its own added vectors, so eight stacks run together), and ranks them. The score is how far the replies move
from the unsteered ones in embedding space, times one minus twice the repeated-trigram rate, plus half of how much
more alike the four replies become than the unsteered four. The last term rewards a character that carries over
from a weekend story to an ocean poem to "Who are you?", which is what a persona looks like, while a neuron that only
inserts its word scores on drift alone.
