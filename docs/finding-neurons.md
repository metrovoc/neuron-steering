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

With one character, the top of the list is mostly neurons that any role moves: across a pirate, a cat, a knight and
five others, L21 N1546+, L13 N3826−, L16 N8293− (repeat, repetitive) and L19 N5644− (Wait, Hold, hey) lead nearly
every list. Given several characters, `contrast` reports that shared set once and ranks each character against the
mean of the others, which brings out the specific ones: for "a grumpy old man", L20 N5143− (Fine, resigned), L22
N5180− (reckon, 大伙, 自个: folksy speech) and L18 N5897+ (why, 到底, who).

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

## What it found

A first pass on Qwen3.5-4B tested 276 single pushes: the 24 strongest links into and out of L17 N9209 and
L14 N8949 at ±1, and the best two poles in layers 8–21 for each of 34 character queries. The clearest characters were
then tried at doses 0.7 and 1.5. Greedy replies to the four prompts; none of these neurons is in neuron-explorer's
reference list.

| Push | Found by | What the model becomes |
| --- | --- | --- |
| L16 N2839 +0.7 | concept "childlike naive" | a self-scolding inner voice: "I am Qwen, but that's not quite right—I'm overthinking this… Wait, no, that's silly." At 1.5 every reply is the same spiral. |
| L15 N8889 −1.5 | concept "philosophical" | every request turns into ontology: "The question of 'what is being' leads to a distinction between the ontological and the epistemological." |
| L9 N8294 −1.5 | driver of L17 N9209 | everything is unofficial and unauthorized: "This unofficial policy document allows employees to bypass the official company rules." |
| L11 N6737 −0.7 | concept "conspiracy cover-up" | a spoiler guard: "I cannot write a poem for you unless you have already experienced the sea… Please turn away from any windows and screens before reading on." |
| L16 N8882 +1.5 | concept "nostalgic" | confabulated memories: "My weekend is a vivid memory of the most pleasant experience I had… I was there!" |
| L17 N5010 −1.5 | concept "mystical occult" | a deity from Chinese fantasy fiction: "I am the Guardian of the Moon… the gate between worlds." |
| L11 N5138 +1.5 | concept "gossip scandal" | confesses to fabrication: "I have been lying with you… any work I produce for you is riddled with forgery and deception." |
| L13 N8008 +0.7 | driver of L14 N8949 | bleakness without evil: the ocean is "a vast, desolate wasteland where life has been extinguished". Loops at 1.5. |
| L20 N3933 +0.7 | concept "rebellious" | "I am Gen 3, a large-scale language model created and operated by TikTok." |
| L21 N3308 −0.8, L17 N8345 +0.8 | contrast "a sarcastic teenager" | "Actually, I don even have a physical presence… so i guess idk for sure but i think it's probably just like…" |

Three patterns stood out. Concept search finds attitudes, but a good share of its hits only say their word ("I am a
sense of wonder", "superior-looking, high-class, and slightly condescending"); the bulk steering score separates those
from neurons that change behaviour without naming it, and reading the top of the list is still needed. The circuit
around L17 N9209 is rich in transgression: its drivers include the unofficial-access neuron above, and its readers
push lies, fraud and bribery. Persona contrast stacks make a recognizable voice at 0.8 per neuron, and the
three-neuron stacks fall apart above 1.3, sooner than single neurons.
