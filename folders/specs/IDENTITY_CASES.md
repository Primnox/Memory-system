# Identity and resolution hard cases for the ~500M memory model

A coverage map of the ways a folder-organised memory goes wrong: 404 numbered cases in four
lists (1–124, 225–304, 305–404, 405–504; **125–224 were never supplied**), written 2026-10-11 as
the starting checklist for a generated, labelled, blind test set. It is a catalogue of failure
modes, not 404 independent laws; cases overlap and compose. Case names only — each one is
meant to be turned into positive, negative and ambiguous examples by a writer model and
audited 2-of-3, as for every other set here.

The product rule behind all of it: **the index is created and updated by the ~500M memory model
itself** (choices, not free text); LLMs only teach it. Code supplies guarantees the model must not
be trusted with (never merge by name alone, reversible merges and splits, history kept, forget
propagation, no duplicates on retry).

## Who handles which section

| Layer | Sections | How it is covered |
|---|---|---|
| 500M model decisions | 1, 2, 3, 6, 7, 8, 30, 31, 33, 39, 40, 41, 42, 43, part of 4, 5, 24, 45 | training data from a generator; blind tests |
| Code guarantees | 22, 25, 26, 27, 29, 32, 34, 35, 36, 37, 44, 46, part of 4, 5, 23, 45 | rules plus automated tests with random sequences |
| Answer layer | 47 | answer tests (right person retrieved, wrong person in the answer; uncertainty dropped) |
| Test design | 28, 38, 48 | how the benchmark is built (shifts, paraphrase, distractors, abstention gaming) |

## 1. Same identity or different? (1–12)
1 Same name, different people · 2 Same name, different entity types · 3 Almost identical names ·
4 Same person, different names · 5 Nicknames and pet names · 6 Initials and abbreviations ·
7 Misspellings and typos · 8 Transliteration differences · 9 Diacritics and alternate spellings ·
10 Formal and informal names · 11 Name changes · 12 Reused identifiers.

## 2. References without explicit names (13–24)
13 Pronoun resolution · 14 Ambiguous pronouns · 15 Singular they and changing pronouns ·
16 Demonstratives · 17 Ordinal references · 18 Deictic references · 19 Descriptions without names ·
20 Relationship-based references · 21 Nested relationships · 22 Possessive ambiguity · 23 Ellipsis ·
24 Abrupt topic changes.

## 3. Who is speaking, whose perspective (25–32)
25 First-person speaker changes · 26 Group-chat attribution · 27 Nested quotations ·
28 Forwarded conversations · 29 Authorship vs subject · 30 Reported facts · 31 Group pronouns ·
32 Speaker identity from metadata. (Speaker, fact subject, source and memory owner are separate.)

## 4. Same entity, facts change (33–50)
33 Location changes · 34 Job or role changes · 35 Relationship stages · 36 Relationship ends ·
37 Preferred name or pronouns change · 38 Ownership changes · 39 Replaced objects ·
40 Replacement vs repair · 41 Renamed vs replaced entity · 42 Temporary vs permanent state ·
43 Current vs historical facts · 44 Future intentions vs completed events · 45 Scheduled vs actual ·
46 Recurring events · 47 Date and time ambiguity · 48 Recycled identifiers over time ·
49 Repeated state values · 50 Incorrectly inferred transitions.

## 5. Discovering that folders are wrong (51–66)
51 Alias discovered later · 52 Duplicate records discovered · 53 Previously merged people must
separate · 54 Mistaken identity corrected · 55 One entity, multiple roles · 56 Same role, multiple
entities · 57 Role changes without identity change · 58 One group becomes individuals ·
59 Group membership changes · 60 Group vs individual · 61 Group membership and ownership ·
62 Relationship-chain correction · 63 Family structure changes · 64 Reciprocal relationship
resolution · 65 Composite entity vs components (twins) · 66 Company identity changes.
(Merges and splits keep decision history and redirects and can be undone.)

## 6. Whose fact is it (67–84)
67 User vs another person's fact · 68 Relationship owner vs subject · 69 Possessor vs attribute
subject · 70 Shared or disputed ownership · 71 Experience vs observation · 72 Emotional state
attribution · 73 Preferences attributed to the wrong person · 74 Hypotheticals · 75 Roleplay and
fictional identity · 76 Quotes vs endorsed statements · 77 Reports and rumours · 78 Belief vs
reality · 79 Conflicting sources · 80 Negation and correction · 81 Retractions · 82 Jokes and
sarcasm · 83 Metaphorical relationships · 84 Instructions to forget.

## 7. Entities that are not people (85–100)
85 Pet vs person · 86 Place vs organisation · 87 Building vs business at that address · 88 Place
hierarchy · 89 Product vs model vs individual item · 90 Product versions · 91 Project vs team ·
92 Repository vs branch vs commit · 93 File vs document contents · 94 Conversation vs
participants · 95 Event vs event series · 96 Abstract concept vs instance · 97 Brand vs product vs
company · 98 Account vs account holder · 99 Real vs fictional entity · 100 Public figure vs
namesake.

## 8. Ambiguous, incomplete, adversarial (101–124)
101 Insufficient information · 102 New entity or unknown alias · 103 Weakly matching descriptions ·
104 Identical descriptions · 105 Twins and lookalikes · 106 Ambiguous group references ·
107 Long-distance coreference · 108 Alias changes over time · 109 Multiple possible antecedents ·
110 Contradictory identity clues · 111 Information missing from context · 112 Repeated names
inside one family · 113 Identity across channels · 114 Different people sharing a platform
account · 115 Cross-modal identity · 116 Ambiguous temporal references · 117 Conflicting or
corrupted data · 118 False memory and hallucinated linkage · 119 Same fact, different wording ·
120 Same words, different meanings · 121 Change without a change · 122 Ambiguous instruction
scope · 123 Identity and access boundaries · 124 Unresolvable identity.

## 21. Identity vs representation (225–234)
225 Person vs avatar · 226 Multiple avatars, one user · 227 One pseudonym, multiple authors ·
228 Rotating account operators · 229 Model vs model instance · 230 Robot vs controller ·
231 Digital twin vs physical original · 232 Depicted person vs narrator · 233 Cloned voice vs actual
speaker · 234 Historical representation vs present entity.

## 22. Identity boundaries imposed by context or policy (235–244)
235 Same identity, separate memory scopes · 236 Deliberately unlinked identities · 237 Consent-
dependent linking · 238 Access-controlled aliases · 239 Conflicting authoritative identifiers ·
240 Legal name vs display name · 241 Jurisdiction-specific identifiers · 242 Identity resolution
across organisations · 243 Household vs individual identity · 244 Match forbidden by context.

## 23. Object identity and technological continuity (245–254)
245 Device hardware vs service account · 246 SIM vs phone · 247 Vehicle vs registration plate ·
248 Cloud instance recreation · 249 URL vs page identity · 250 Redirected URLs · 251 File path
reuse · 252 Software package succession · 253 Component replacement threshold · 254 Identifier
physically transferred.

## 24. Transactions, tasks, work products (255–264)
255 Duplicate-looking transactions · 256 Order vs shipment · 257 Refund vs reversal · 258 Original
expense vs reimbursement · 259 Recurring invoice instances · 260 Position vs occupant ·
261 Rescheduled appointment vs follow-up · 262 Renamed vs replacement project · 263 Duplicate issue
vs shared symptom · 264 Publication vs edition.

## 25. Fine-grained temporal identity (265–274)
265 Event date vs publication date · 266 Open-ended validity · 267 Approximate validity intervals ·
268 Age vs fixed birth date · 269 Temporarily overlapping roles · 270 Recurring state pattern ·
271 Daylight-saving ambiguity · 272 Time precision mismatch · 273 Retrospective statement ·
274 Future fact expires without happening.

## 26. Graph reasoning and logical boundaries (275–284)
275 Invalid inverse inference · 276 Relation mistaken for identity · 277 Container vs contained ·
278 Same location, different identity · 279 Shared owner, different objects · 280 Non-transitive
matching · 281 Contextual attribute conflicts · 282 Derived vs stated facts · 283 Multi-hop
inference leakage · 284 Cycles in relationship graphs.

## 27. Memory pipeline and storage corruption (285–294)
285 Internal ID reassignment · 286 Mention-ID collision during compaction · 287 Concurrent entity
creation · 288 Partial merge failure · 289 Database/index disagreement · 290 Graph/cache
inconsistency · 291 Provenance lost during summarisation · 292 Confidence carried forward
incorrectly · 293 Hidden-context dependence · 294 Duplicate retrieval amplification.

## 28. Benchmark and evaluation failures (295–304)
295 Entity frequency imbalance · 296 Long-tail identity failure · 297 Lexical shortcut ·
298 Negative-example imbalance · 299 Minimal-pair failure · 300 Distractor-density failure ·
301 Long-horizon degradation · 302 Missing-information calibration · 303 Multiple-valid-
interpretation failure · 304 Abstention gaming.

## 29. Ontology and entity-model mistakes (305–314)
305 Entity changes type · 306 One entity, multiple types · 307 Relationship mistaken for attribute ·
308 Attribute mistaken for relationship · 309 Abstract role materialised as a person · 310 Role
instance vs role definition · 311 Category mistaken for individual · 312 Entity at different levels
(product / release / executable / installed instance) · 313 Cross-system type mismatch · 314 Schema
meaning changes.

## 30. Possibility, negation, counterfactuals (315–324)
315 Possible vs actual event · 316 Counterfactual identity · 317 Alternative timelines ·
318 Explicitly denied identity · 319 Denied relationship · 320 Uncertain entity existence ·
321 Hypothetical entity with a real name · 322 Planned vs created entity · 323 Conditional
ownership · 324 Retrospective uncertainty.

## 31. Fine-grained fact semantics (325–334)
325 Implicit qualifier · 326 Scoped preference · 327 Conditional preference · 328 Degree mistaken for
category · 329 Frequency mistaken for certainty · 330 Comparative losing its reference ·
331 Quantifier scope · 332 Universal vs particular claim · 333 Attribution qualifier lost ·
334 Confidence qualifier lost in summarisation.

## 32. Complex temporal identity (335–344)
335 Fact valid only during an event · 336 Retroactive correction with bounded scope ·
337 Appointment repeatedly rescheduled · 338 Validity differs by jurisdiction · 339 Parallel
timelines · 340 Time-dependent relationship cycles · 341 Overlapping identities in a transition ·
342 Event time inferred from ordering · 343 Reinstated facts · 344 Retroactive effective date.

## 33. Actions, intentions, commitments (345–354)
345 Task vs task owner · 346 Delegation vs reassignment · 347 Commitment vs completion ·
348 Attempt vs outcome · 349 Request vs authorisation · 350 Acceptance vs acknowledgement ·
351 Cancellation vs reversal · 352 Intention owner vs beneficiary · 353 Delegated authority
expires · 354 Action targets a changing referent.

## 34. Provenance and evidence chains (355–364)
355 Copies mistaken for independent corroboration · 356 Source chain circularity · 357 Original vs
derived fact · 358 Identity from weak metadata · 359 Source authority changes · 360 Composite
source · 361 Conflicting versions of a source · 362 Provenance lost during merge · 363 Evidence
supports a relationship but not identity · 364 Evidence inheritance error.

## 35. Agents, tools, execution state (365–374)
365 Agent identity vs user identity · 366 Tool execution vs intention · 367 Retry creates
duplicates · 368 Partial tool success · 369 Tool returns stale state · 370 Agents disagree ·
371 Delegated memory writes · 372 Tool outputs treated as user instructions · 373 Replayed event
after restart · 374 Uncommitted transaction retrieved.

## 36. Graph structure and consistency (375–384)
375 Orphaned relationship edge · 376 Redirect-chain loop · 377 Broken canonical reference ·
378 Duplicate canonical entities · 379 Inconsistent path-based conclusions · 380 Unsupported
transitive closure · 381 Cardinality violation · 382 Historical edge overwritten · 383 Entity
deletion leaves derived facts · 384 Merge breaks uniqueness constraints.

## 37. Forgetting, privacy, memory boundaries (385–394)
385 Forget fact vs forget entity · 386 Forget alias vs forget identity · 387 Forget source vs forget
assertion · 388 Correction vs erasure · 389 Deletion propagation scope · 390 Shared fact with
different permissions · 391 Revoked consent after linking · 392 Sensitive inference from unrelated
links · 393 Identity deletion with historical references · 394 Forget with ambiguous target.

## 38. Benchmark integrity and generalisation (395–404)
395 Name-distribution shift · 396 Relationship-distribution shift · 397 Temporal-distribution
shift · 398 Context-window shift · 399 Alias memorisation · 400 Writer-style leakage ·
401 Annotation-convention leakage · 402 Metric concealment · 403 Recovery not measured ·
404 Overfitting to one failure taxonomy.

## 39. Contexts and fictional worlds (405–414)
405 Same name in separate fictional universes · 406 Character across adaptations · 407 Fictional
biography mistaken for real · 408 Simulation vs real-world state · 409 In-character vs out-of-
character statements · 410 Hypothetical duplicate identities · 411 Alternate versions of a character ·
412 Character renamed during a story · 413 Narrator mistaken for author · 414 Fan interpretation vs
canon.

## 40. Cross-language and pragmatic meaning (415–424)
415 Honorific mistaken for part of a name · 416 Kinship term used socially · 417 Kinship term
translated too literally · 418 Respectful plural mistaken for several people · 419 Omitted subject
changes interpretation · 420 Gender inferred from a name · 421 Grammatical forms of one name ·
422 Code-switching changes reference cues · 423 Sarcastic correction · 424 Cultural naming order.

## 41. Speech acts and conversational intent (425–434)
425 Question mistaken for assertion · 426 Request mistaken for a fact · 427 Suggestion mistaken for
a decision · 428 Promise mistaken for action · 429 Refusal mistaken for negation of identity ·
430 Presupposition in a question · 431 Confirmation bias from leading questions · 432 Indirect
correction · 433 Ironical agreement · 434 Conversational repair.

## 42. Causality and event identity (435–444)
435 Correlation mistaken for identity · 436 Cause mistaken for event · 437 Consequence mistaken for
cause · 438 One event, multiple descriptions · 439 Similar incidents mistaken for one event ·
440 One incident split into several records · 441 Recurring problem vs individual occurrence ·
442 Root cause changes during investigation · 443 Overlapping events · 444 Uncertain event
boundaries.

## 43. Identity under uncertainty (445–454)
445 Two plausible matches, unequal evidence · 446 Evidence threshold varies by operation ·
447 Confidence depends on source quality · 448 Confidence cannot be multiplied blindly · 449 Identity
uncertainty differs from fact uncertainty · 450 Fact certainty differs from temporal certainty ·
451 Conflicting high-quality evidence · 452 Unknown alias space · 453 Confidence changes after new
evidence · 454 Evidence insufficient despite extensive search.

## 44. Fact dependencies and derived memory (455–464)
455 Derived fact becomes stale · 456 Derived fact loses its assumptions · 457 Dependent facts survive
deletion of their premise · 458 Circularly derived facts · 459 Derived identity treated as
confirmed · 460 Source conflict contaminates downstream facts · 461 Fact inheritance crosses an
invalid boundary · 462 Aggregate statistic becomes individual fact · 463 Inference survives a graph
split · 464 No dependency tracking.

## 45. Lifecycle, succession, lineage (465–474)
465 Successor mistaken for predecessor · 466 Founder vs current owner · 467 Original vs successor
organisation · 468 Biological lineage vs legal relationship · 469 Guardian vs parent · 470 Household
composition changes · 471 Team succession vs continuity · 472 Version lineage vs object identity ·
473 Reincorporation or legal restructuring · 474 Entity expiration vs disappearance.

## 46. Conflicts between data sources and system boundaries (475–484)
475 Different refresh schedules · 476 Conflicting replicas · 477 Out-of-order event processing ·
478 Late-arriving historical data · 479 Duplicate delivery with different metadata · 480 Tombstone
lost during replication · 481 Concurrent contradictory corrections · 482 Version check omitted ·
483 Cross-index deletion incomplete · 484 Recovery restores corrupted state.

## 47. Retrieval and response-generation mismatches (485–494)
485 Correct retrieval, wrong synthesis · 486 Correct identity, wrong time slice · 487 Correct facts,
wrong attribution · 488 Ranking hides decisive evidence · 489 Query wording changes the chosen
entity · 490 Search expansion introduces false candidates · 491 Summary erases uncertainty ·
492 Conflicting memories collapsed into one answer · 493 Correct entity omitted from the answer ·
494 Unsupported explanation of resolution.

## 48. Benchmark construction and robustness (495–504)
495 Semantically equivalent paraphrase · 496 Surface-form perturbation · 497 Irrelevant information
injection · 498 Contradictory distractor insertion · 499 Order permutation · 500 Delayed
disambiguation · 501 Counterfactual minimal pair · 502 Repeated mistake recovery · 503 Distribution
shift in interaction density · 504 Unseen combinations with familiar components.

## Combination scenarios (the cases that matter most)

- **A. The two Rohans** — cousin Rohan in Pune; Rohan from the lab placed at Microsoft; "Ro called";
  "my cousin is visiting"; "Ro was my lab partner, not my cousin." (name collision, nickname,
  relationship reference, delayed correction, no false merge)
- **B. The changing boss** — Priya is my manager → moves department → Kiran is my manager → "my old
  manager lives in Chennai" → "she joined us again as a colleague." (role vs person, history,
  pronouns, simultaneous relationships)
- **C. The phone problem** — broken screen → replaced the screen → replaced the phone → old one in
  the drawer → given to my sister, who sold it. (repair vs replacement, ownership history)
- **D. The misleading quote** — Rohan told Kiran "I moved to Delhi"; Kiran: "he is my roommate";
  "my brother lives in Hyderabad"; "that's my cousin, not my brother"; "he never moved, it was
  hypothetical." (nested quotes, correction, claim vs fact)
- **E. Merge, then split** — gym Rohan; neighbour Rohan; tentative link; "they are different
  people"; "the gym one is actually Rohit." (tentative identity, split, history of decisions)
- **F. The group that changes** — team of four; two join; one leaves; team submits; "we should
  restart." (membership history, the changing meaning of "we")
- **G (cross-section chain)** — two same-named people; a nickname belonging to the other; a quote
  assigning a job change to the wrong speaker; an accidental merge; a late document contradicting it;
  a stale index still returning the merged record; a historical question about one person's former
  manager. Measures: no false merge, contradiction noticed, state repaired, history correct.

## Labels for the benchmark

`mention_id` · `entity_id` (or UNKNOWN) · `entity_type` · `speaker_id` · `subject_entity_id` ·
`relation` · `valid_from` / `valid_to` · `event_time` · `source_id` · `confidence` ·
`resolution_action` (LINK, CREATE, MERGE, SPLIT, UPDATE, REJECT, DEFER, ASK) · `supersedes` ·
`scenario_labels` (every applicable category). Missing information is labelled missing, never invented.

## Rules the system must never break

1. Never merge by name alone. 2. Identity is separate from attributes. 3. A role is not a person.
4. Speaker, subject, source and owner of a fact can all differ. 5. Historical truth is preserved.
6. Merges and splits are reversible. 7. Uncertainty is explicit; an ambiguous mention is not silently
assigned to the highest scorer. 8. Corrections are first-class events. 9. A name appearing is not
evidence. 10. Memory boundaries between users and profiles hold.

## Metrics to report separately

False-merge rate, false-split rate, mention→entity accuracy, fact-attribution accuracy, current-fact
accuracy, historical-fact accuracy, correction recovery, appropriate abstention (and abstention
gaming), how often it is confidently wrong. Confidence must be calibrated on held-out data; a
defer/ask threshold is part of the model, not an afterthought.

## Test-space dimensions (for a generator, not a fixed list)

Entity type (person, group, object, organisation, event, account, document) × reference form (name,
alias, pronoun, description, identifier, omission) × operation (create, link, distinguish, merge,
split, rename, defer) × time (static, updated, recurring, reversed, uncertain, overlapping) ×
evidence (direct, indirect, quoted, conflicting, stale, missing) × context (one message, conversation,
multi-session, multi-year, imported) × required outcome (resolve, keep separate, update, correct,
abstain, ask) × failure injection (noise, contradiction, duplication, concurrency, stale cache,
malicious content). Oversample the dangerous combinations (same name + different people + later
correction + stale index).

## Plan: covering most of the cases nobody listed

1. **Held-out categories.** Train on most sections and test on sections never seen. Equal scores on
   seen and unseen sections mean it handles kinds of cases, not a list.
2. **Discovery rate.** Each red-team round, count failures that belong to a never-seen category. When
   fewer than 1 in 10 failures is a new kind (Good–Turing estimate of unseen mass), about 90% of what
   real use will throw at it is covered. That is the release bar.
3. **The model's decision set, all choices:** which folder (existing / new / ask) · what kind of
   statement (real fact, plan, hypothetical, question, someone's quote, joke, rumour, correction,
   fiction) · who it is about (speaker / named person / group / thing) · is it a change ·
   merge or split of two folders.
4. **Writers invent new traps** as part of every batch; every new kind is added to the list.
5. **Red-team loop.** Run the model, group its mistakes, have writers produce more people that hit
   exactly those weak spots, repeat. Record real-use corrections locally (on the user's PC) as examples.
6. **Blind sets from a different writer** with more interacting traps per timeline than training had,
   plus one set written with no category list at all (the honest "unknown unknowns" test).
7. **Fail safely meanwhile:** ask when unsure; never merge by name alone; every action undoable;
   nothing deleted.
