# Folder memory: folders, slots, an index — and the head together

Work of 2026-10-08. Numbers are copied from the files in `results/`.

## The design

Memory organised the way a person keeps an organiser, instead of one flat pile of facts:

- **Folders** — one per person, pet, place, organisation, thing or activity in someone's
  life, plus one per speaker. A folder has a name, aliases ("my sister", "sis", "Maya's
  sister") and how it links to the speaker.
- **Slots** — every fact is filed under one attribute of one folder: `Sarah.lives_in`,
  `me.employer`, `Biscuit.age`. A slot holds ONE value at a time (where someone lives, their
  phone) or MANY (hobbies, trips, events).
- **An index** — a main index of folders, a days index (what was said on which day) and a
  card per folder ("[Sarah | my sister] lives in: …, job: …").
- **Everything is linked** — a fact sits in every folder it touches and on its day, so the
  folders form a web: "who did I meet yesterday" walks days → that day → the person's folder.

Facts are filed by a "clerk" model (`specs/SPEC_FOLDERS.md`) in date order, 25 at a time, each
call seeing only the folders and slots filed so far — never later facts, never any label.
The clerk here is Gemini 3.8 Flash (`pipeline/folders.py`); an open-model clerk (Qwen 2.5 7B,
`kaggle/primnox-folders-qwen`) is the local version, still being scored.

## Results

Settings were chosen on the **dev set** (the 24 round-7 chat people; head training data) and
fixed before any proof set was scored. Proof sets were scored once.

### Search on blind v2 (proof set, 171 questions)

| How memory is searched | Right fact first | In the top 3 |
|---|---|---|
| Plain similarity (gte-small), nothing retired | 41.5% | 85.4% |
| Folder cards: each fact searched as "Sarah (my sister) \| lives in \| …" | 47.4% | 92.4% |
| Folders + index, **no head** (days index; one-value slots rank older values down) | 73.7% | 95.9% |
| **Today's app**: round-7 head retirements + dates read from the question + plain similarity | 80.1% | 94.2% |
| **Today's app + folder cards** | **86.0%** | **97.7%** |
| Ceiling: a perfect head (from the labels) + folder cards | 88.3% | 97.7% |

Dev set, same pipeline: today's app 74.1% → **80.9%** first (91.5% top 3). On dev "back then"
questions 57.7% → 69.1%, from two rules chosen on dev: a named period is searched as of its
end, and a question dated by an event ("before I bought the iPhone 14", "back when I was at
LAAS") takes that event's date (`pipeline/real_pipeline.py`).

What this shows: folders are not a replacement for the change-detection head — a perfect head
alone beats folders alone (82.5% vs 73.7%, `results/oracle_app_blind_and_dev.txt`). They do
add on top of it: +5.9 points first and +3.5 in the top 3 with the real round-7 head, which
here is within about 2 points of a perfect one (104/112 changes, 0 wrong).

### Slots and change detection

- The fact a new one replaced was filed in the **same slot** 82% of the time on blind v2
  (92/112).
- Giving the head every fact in the new fact's slot as extra candidates changed nothing on
  blind v2 or dev (`results/head_sweep_scores.txt`): with 25–65 facts per person, today's 15
  most similar + 12 name-linked candidates already reach nearly everything. It is meant for
  long histories, which these sets do not have.

### Real chats (REALTALK, 10 chats, ~300 facts each, 505 questions)

| Search over the Gemini-extracted facts | First | Top 3 |
|---|---|---|
| Plain similarity | 45.7% | 66.1% |
| + entity overlap with the question | 45.3% | 66.5% |
| + one hop through shared entities | 38.2% | 65.1% |
| Plain + round-7 head retirements | 41.6% | 58.0% |
| Plain + head, retirements only between facts about the same person | 42.0% | 58.4% |

- One hop helps multi-hop questions a little (30.9 → 33.6% first) and hurts time questions
  badly (67.6 → 48.8%): hops must be used only when the question is about connections.
- **Retiring facts hurts on real chats.** Most REALTALK questions are about history ("what did
  she make on 29 December", "how many times has she been to …"). The app hides retired facts
  from search; for 37 time questions every answer fact had been retired. Only 32 of the 435
  retirements were between facts about different people. Next: keep retired facts and let the
  question decide — "now" questions skip them, dated questions search as of then, history
  questions see everything. Not measured yet.
- REALTALK has no licence: none of its messages, facts or filings are in this repo.

### The web walk (what the model reads)

At the same size budget, walking the web (top facts + cards of the folders they touch + cards
one link away) carried about as many answer facts as plain top results on the dev people
(`results/folder_walk_dev_chat.txt`) — their memories are so small that anything fits. It
needs long histories and a test of a model answering from it.

### Extractor v2 (a negative result)

Retraining the local 7B extractor to stop skipping short messages (data:
`pipeline/build_extractor_data_v2.py`; job: `kaggle/primnox-extractor-v2`) made it worse on
LongMemEval knowledge updates, with the round-7 head:

| Facts written by | F1 | Precision | Recall |
|---|---|---|---|
| Gemini 3.8 Flash (prompt) | 0.78 | 0.77 | 0.79 |
| Qwen 2.5 7B, v1 (round 7) | **0.67** | 0.78 | 0.58 |
| Qwen 2.5 7B, v2, 2 passes | 0.60 | 0.75 | 0.50 |
| Qwen 2.5 7B, v2, 1 pass | 0.56 | 0.76 | 0.44 |

It did write facts for more messages (updates with an empty side: 17 → 4 of 72), but learned to
copy whole short messages back as "facts" (35 times). The training data is not published: it
holds statements from the private v3 set. Next: train the extractor to imitate the cloud
model's own extractions on non-benchmark chats.

## Files

| Path | What |
|---|---|
| `specs/SPEC_FOLDERS.md` | the clerk's filing rules |
| `pipeline/folders.py` | the Gemini clerk (agy; uses `round7/pipeline/drive.py`) |
| `pipeline/folder_eval.py` | search A / F / A+R / F+R; change candidates SIM vs SLOT |
| `pipeline/folder_index.py` | the index walk (days, one-value slots) |
| `pipeline/folder_walk.py` | what the model reads, at a size budget |
| `pipeline/oracle_app.py` | the app's approach with a perfect head, beside the index |
| `pipeline/real_pipeline.py` | the real head's retirements + dates read by code + folder cards |
| `pipeline/index_misses.py` | why the index misses (dev only) |
| `pipeline/graph_search.py` | REALTALK entity / one-hop test |
| `kaggle/primnox-head-sweep` | the round-7 head run as the app runs it, with and without slot candidates |
| `kaggle/primnox-folders-qwen` | the open-model clerk (Qwen 2.5 7B, 4-bit, both T4s) |
| `data/folders_gemini_{chat,blind}.json` | the Gemini clerk's filing of the public sets |
| `data/lme_facts/` | extractor v2's facts for LongMemEval's 467 messages |

Dates are read with the memory's own date reader, `backend/primnox2/memory/when.py`. The
REALTALK scripts expect the facts file built by `round7/pipeline/realtalk_e2e.py` from a local
REALTALK clone.

## Rules kept

- Blind v2, LongMemEval and REALTALK are proof sets: never trained or tuned on. Every setting
  here (ALPHA, TIME, STALE, the dated-question rule) was chosen on the dev set.
- The dev set (round-7 chat people) and v3 are head training data; head scores on them are
  not proof.
