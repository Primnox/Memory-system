# graph — a graph over memories as a prompt index (experiment, 2026-10-04)

Question: instead of sending every current fact with every message (up to 200,
~3–4k tokens), can a graph of facts linked to the people, pets, places, things
and topics they mention pick the few facts a message needs?

`graph_index.py` builds the graph without any language model (names from
capitalisation and possessives, topics from Primnox's word lexicon, content
words weighted by rarity) and ranks facts with a personalized PageRank walk
seeded by the message's own names and topic and its 3 most similar facts (the
HippoRAG idea). `eval_block.py` checks, for every "now" and multi-value
question, whether the chosen block still contains all answer facts, and
estimates tokens. Live facts come from the labels, so this measures selection
alone. `--pad 200` adds facts from other people as noise (never safety facts)
to reach the app's 200-fact cap; it is a harsh test, since the noise includes
strangers' "I live in…" facts.

Developed on the dev sets (64 questions), then scored once on blind `v2`
(117 questions):

| v2, 200-fact stores | Answer in block | Tokens |
|---|---|---|
| Everything (today) | 100% | 4,075 |
| Similarity, top 20 | 79% | 454 |
| Graph, top 20 | 68% | 467 |
| Hybrid (rank fusion), top 20 | 79% | 463 |

| v2, real size (~16 facts) | Answer in block | Tokens |
|---|---|---|
| Everything | 100% | 384 |
| Similarity, top 8 | 97% | 213 |
| Graph, top 8 | 93% | 214 |
| Hybrid, top 8 | 96% | 214 |

**Finding:** choosing facts cuts prompt tokens by roughly 45% (small stores)
to 89% (200 facts), but this graph does not choose better than plain
similarity; at 200 facts it is worse. The saving comes from selection, not
from the graph. Selection also costs answers at scale (79% kept), so a
trimmed block needs the `recall_memory` tool as a fallback.

Why the graph did not help here: the test questions are paraphrased and
rarely name the person or thing they ask about; rule-based names and keywords
connect noise facts through common words; and there are few multi-hop
questions ("what does my sister's husband do?"), which is where a graph should
win. Next tries, if any: a learned entity linker (the small model), edges
between people ("Tom" ↔ "my boyfriend"), and a set with multi-hop questions.
The graph may still pay off elsewhere: as candidates for change detection
(chain reactions), where shared names already raised reach from 72% to 77%.

Results: `results/block_*.json`.
