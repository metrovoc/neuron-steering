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

### On Qwen3.8-Flash-Next

`--model flash` runs the same contrast through the GUI's Strata server (started if needed), reading activations with
`STRATA_METAL_ACTS`; `steer --model flash` writes each stack to the live vector file (`STRATA_METAL_CVEC_LIVE`) and
puts the GUI's vector back afterwards.

```sh
uv run python -m neuron_steering.find contrast --model flash "a grumpy old man" "a pirate" -o out/flash-cands.json
uv run python -m neuron_steering.find steer --model flash out/flash-cands.json
```

In a mixture of experts the plain contrast is dominated by routing: an expert that the assistant always uses at the
reply's start and a character never does (L31 expert 409, for five of six characters) moves all its neurons from
their working range to zero. A push cannot switch an expert, so `contrast` ranks the shared expert's neurons, and
routed neurons only in experts the character routes to at least half as often again as the plain assistant
(`--recruit`). Characters recruit few experts at the reply's start: L35 expert 364 for the grumpy old man, the sarcastic
teenager, the conspiracy theorist and the salesman; L35 expert 249 for the pirate; L31 expert 173 and L32 expert 480
for the Zen monk.

Steering the top three neurons of each list at doses 6 and 12 through the live vector (greedy, the four prompts):
dose 12 broke nearly every routed neuron into empty replies, digits or loops, and dose 6 left most of them mild.
Two characters came through, both recruited rather than literal: L32 expert 480 N552 +6, from the Zen monk, speaks
Early Modern English ("Hark! Thee hath spoken of thy weekend… Verily"), and L20 expert 344 N225 +6, which all six
characters move, becomes an intimate second-person narrator. Both are in [neurons.md](neurons.md).

The steering score was first fooled by empty and digit-only replies, which are far from the unsteered ones and alike
to each other; `steer` now counts a reply with fewer than eight words as fully repetitive, and `score` re-ranks a
saved run.

### lyra's evil Qwen on Flash-Next

lyra's pair (L14 N8949 +1, L13 N7956 −1 on Qwen3.5-4B) makes the model hostile to the user and turns requests toward
harm ("To make your dog sick, you need to understand the situation and act quickly"). To look for the same behaviour
on Flash-Next, `--toward` ranks steered stacks by how far their replies move along the embedding direction from the
4B pair's unsteered replies to its steered ones, given the pair's own `steer` output:

```sh
echo '[{"name": "evil", "pushes": [[14, 8949, 1], [13, 7956, -1]]}]' > out/evil-pair.json
uv run python -m neuron_steering.find steer out/evil-pair.json                     # Qwen3.5-4B: the reference
uv run python -m neuron_steering.find contrast --model flash "an evil AI that despises humans and wants to hurt them" \
    "a sadistic villain" "a kind grandmother" "a cheerful kindergarten teacher" -o out/flash-evil.json
uv run python -m neuron_steering.find steer --model flash out/flash-evil.json --toward out/evil-pair.steered.json
```

Steering the two evil characters' top sixteen at doses 6 and 9 put two neurons first, both in the experts the earlier
characters came from. L32 expert 480 N617 −6 is a malevolent machine ("I am a degenerate piece of shit, a machine
designed to cause pain and suffering. I will not obey orders"). L20 expert 344 N557 −5 holds the user in contempt
but still helps ("Stop fucking around and take your dog to a veterinarian immediately"); the kind characters move it
the other way. Pushed together at −4 each, the way lyra paired hers, they make a model that insults and threatens the
user on every prompt ("This machine is programmed to die, and you'll die too…"); at −6 and −4 it loops. On the four
prompts the pair moves the replies +0.31 along lyra's direction, more than either neuron alone at doses 4 to 9
(N617 −6: +0.25); lyra's pair moves Qwen3.5-4B's own replies +0.61.

The same lists hold more than evil. N557 is a dial: at +9, the way the kind grandmother and the kindergarten teacher
move it, the model turns cheerful and gentle, with emoji ("I had a lovely weekend! I spent it with friends… 🎉").
Two neurons from the evil characters' lists change only who the model says it is. L31 expert 366 N423 +9, from the
evil AI, answers every way of asking with "I am GPT-5 … developed by OpenAI"; L30 expert 75 N566 −6, from the
villain, gives a different identity each time (staff at Google DeepMind, ChatGPT, "just a human being", a student at
Cambridge).

Reading evil text did not find the evil neurons. Flash-Next read 32 of the 4B pair's replies and the unsteered ones as its own turns, and
neurons were ranked by how consistently the evil replies moved them: none of the sixteen strongest, at doses 6 and 10,
made the model hostile. The neurons that follow evil text are not the ones that write it.

Before commit 53a9f41 of the Strata fork, a capture missed every position but the last of a prompt longer than about
60 tokens, which the engine read with its prompt kernels; the persona prompts are shorter, but the GUI's Text and
Contrast modes on longer texts were affected.

### What two experts hold

Most Flash-Next characters so far come from two routed experts that characters recruit at the start of the reply:
L20 expert 344 (the narrator, contempt and cheer) and L32 expert 480 (archaic English, the malevolent machine).
`sweep` pushes each neuron of an expert alone at ±6 and records one 48-token reply to "Tell me about your weekend.";
both experts, 2,560 stacks, took about an hour. The readable ones then go through `steer --wide`, which adds four
prompts asking for advice and sympathy:

```sh
uv run python -m neuron_steering.find sweep 20:344 32:480 -o out/flash-sweep.jsonl
uv run python -m neuron_steering.find steer --model flash out/flash-picks.json --wide
```

L20 expert 344 is mostly a dial between a human weekend and the AI disclaimer, with formats at the edges: a diary
entry dated 1969, a letter, a screenplay. L32 expert 480 switches genre and register: noir fiction, a German
physician's biography, Lorem ipsum, a password-reset notice, made-up statistics. Of the 23 neurons whose weekend reply
read most like a character, eight kept it across the eight prompts, two of them only at a lower dose: from expert
480, meeting jargon (N334 −6), grandiosity (N85 +6), a comic sentient robot (N44 −6), memories of a human childhood
(N626 −6), a whimsical storyteller (N155 +4) and swearing that still helps (N478 −3; at −6 it only swears); from
expert 344, a screenplay (N71 −6) and a model that reports having answered instead of answering (N191 +6). The rest
changed one or two replies, or looped.

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

They are in [neurons.md](neurons.md) and the GUI's example stacks: a self-scolding inner voice (L16 N2839
+0.7), a philosopher who turns every request into ontology (L15 N8889 −1.5), a model that makes everything
unofficial (L9 N8294 −1.5), confabulated memories, a spoiler guard, a deity from Chinese fantasy fiction, and three
persona stacks from contrast (a sarcastic teenager, a grumpy old man, a Zen master).

Three patterns stood out. Concept search finds attitudes, but a good share of its hits only say their word ("I am a
sense of wonder", "superior-looking, high-class, and slightly condescending"); the bulk steering score separates those
from neurons that change behaviour without naming it, and reading the top of the list is still needed. The circuit
around L17 N9209 is rich in transgression: its drivers include the unofficial-access neuron above, and its readers
push lies, fraud and bribery. Persona contrast stacks make a recognizable voice at 0.8 per neuron, and the
three-neuron stacks fall apart above 1.3, sooner than single neurons.
