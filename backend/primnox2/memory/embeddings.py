"""Meaning-based matching for permanent memory. Optional, and off by default.

`memory/service.py` ranks by words and a hand-written topic lexicon, so
"what pets do I have" finds nothing in "My dog is called Biscuit" — no shared
word, no alias. This adds the missing signal from the sentence encoder that
`tools/retrieval.py` already loads. Two independent uses, each behind its own
tunable (`memory.embeddings`, `memory.embeddings_supersede`):

  SEARCH       cosine similarity is fused with the lexical order (see `fuse`).
  SUPERSESSION a new statement's similarity to each live memory is offered to
               `cognition/conflict.py` as one more way to find "same slot,
               different value", behind the structural guards it already has.

The write path never needs a model. Every function here returns None when the
encoder is not up (still loading, absent, or it raised), and the caller then
does exactly what it did before this module existed. Vectors are cached in the
`memories.embedding` column — a V1 leftover nothing else writes — as
`8-byte tag + float32[]`. The tag is a hash of (encoder, text), so an edited
memory, or a different encoder, reads as "not cached" with no invalidation
step to forget; `service.update()` also clears the column so a stale vector is
not carried around.

Rows handed to callers never include the blob (`service.live()` strips it): it
is not JSON and nothing outside this module can use it.
"""
from __future__ import annotations

import hashlib
import logging
import math
import threading
from array import array
from operator import mul

from ..storage import db

log = logging.getLogger(__name__)

# Reciprocal-rank fusion's customary constant. Not a tunable: it only decides
# how fast a rank's weight falls off, and every value in the usual 10-100 range
# orders a hundred memories the same way.
RRF_K = 60

# Encoder batch. The encoder pads to the longest text in a batch, so a large
# one spends its time on padding and a small one on call overhead.
BATCH = 64

_TAG = 8

# The most texts a caller will encode itself. A store that was never embedded has
# every row uncached, and encoding hundreds inline stalls the first write or search
# after the encoder comes up for seconds. A bigger gap is filled on a thread and the
# caller answers with words alone until it is.
SYNC_LIMIT = 64
_filling = threading.Lock()


def prepare(wait_s: float = 0.0) -> bool:
    """Bring the encoder up, optionally waiting for it. For benchmarks and
    tests that want the cold start paid before anything is timed; the app
    leaves it to load in the background and answers lexically until it does."""
    from ..tools import retrieval
    return retrieval.encoder(wait_s) is not None


def available() -> bool:
    from ..tools import retrieval
    return retrieval.encoder() is not None


def _unit(vec: array) -> array:
    norm = math.sqrt(sum(map(mul, vec, vec)))
    return vec if not norm else array("f", (x / norm for x in vec))


def cosine(a: array, b: array) -> float:
    """Dot product — every stored vector is unit length."""
    return sum(map(mul, a, b))


def encode(texts: list[str]) -> list[array] | None:
    """Unit vectors for `texts`, or None when there is no encoder to ask."""
    from ..tools import retrieval
    fn = retrieval.encoder()
    if fn is None:
        return None
    try:
        out: list[array] = []
        for i in range(0, len(texts), BATCH):
            raw = fn(texts[i:i + BATCH])
            rows = raw.tolist() if hasattr(raw, "tolist") else raw
            out.extend(_unit(array("f", row)) for row in rows)
        return out
    except Exception:
        # Losing the encoder costs the semantic signal, never the search.
        log.warning("memory embeddings: encoder failed, using lexical ranking", exc_info=True)
        return None


def _tag(text: str) -> bytes:
    from ..tools import retrieval
    return hashlib.sha1(f"{retrieval.ENCODER}\0{text}".encode("utf-8")).digest()[:_TAG]


def pack(text: str, vec: array) -> bytes:
    return _tag(text) + vec.tobytes()


def _unpack(blob: bytes | None, tag: bytes) -> array | None:
    if not blob or blob[:_TAG] != tag or (len(blob) - _TAG) % 4:
        return None
    vec = array("f")
    vec.frombytes(blob[_TAG:])
    return vec


def _cached(rows: list[dict]) -> tuple[dict[str, array], list[dict]]:
    blobs = {r["id"]: r["embedding"] for r in db.connect().execute(
        "SELECT id, embedding FROM memories"
        " WHERE embedding IS NOT NULL AND deleted_at IS NULL")}
    found: dict[str, array] = {}
    missing: list[dict] = []
    for r in rows:
        vec = _unpack(blobs.get(r["id"]), _tag(r["text"]))
        if vec is None:
            missing.append(r)
        else:
            found[r["id"]] = vec
    return found, missing


def vectors(rows: list[dict]) -> dict[str, array] | None:
    """A unit vector per row id: cached where current, embedded (and cached)
    where not. None when something needs embedding and there is no encoder, or
    when too much does to wait for (see SYNC_LIMIT)."""
    if not rows:
        return {}
    found, missing = _cached(rows)
    if missing:
        if len(missing) > SYNC_LIMIT:
            _fill_later()
            return None
        fresh = encode([r["text"] for r in missing])
        if fresh is None:
            return None
        for r, vec in zip(missing, fresh):
            found[r["id"]] = vec
        _store([(pack(r["text"], v), r["id"], r["text"]) for r, v in zip(missing, fresh)])
    return found


def fill() -> int:
    """Embed and cache every standing memory that has no current vector, a batch
    at a time. Returns how many were done; stops at the first batch the encoder
    cannot do."""
    rows = [{"id": r["id"], "text": r["text"]} for r in db.connect().execute(
        "SELECT id, text FROM memories WHERE deleted_at IS NULL")]
    todo = _cached(rows)[1]
    done = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        fresh = encode([r["text"] for r in chunk])
        if fresh is None:
            break
        _store([(pack(r["text"], v), r["id"], r["text"]) for r, v in zip(chunk, fresh)])
        done += len(chunk)
    return done


def _fill_later() -> None:
    if not _filling.acquire(blocking=False):
        return

    def work() -> None:
        try:
            fill()
        except Exception:
            log.warning("memory embeddings: background fill failed", exc_info=True)
        finally:
            _filling.release()

    threading.Thread(target=work, name="memory-embeddings", daemon=True).start()


def _store(items: list[tuple]) -> None:
    # `AND text=?` so a vector computed for the old wording cannot be filed
    # under the new one if an edit lands between the read and this write.
    # `updated_at` is left alone: a cache fill is not a change to the memory.
    try:
        with db.tx() as c:
            c.executemany("UPDATE memories SET embedding=? WHERE id=? AND text=?", items)
    except Exception:
        log.warning("memory embeddings: could not cache vectors", exc_info=True)


def rank(query: str, rows: list[dict], *, min_similarity: float) -> list[str] | None:
    """Ids of `rows` by similarity to `query`, best first, keeping only those at
    or above `min_similarity`. None when the encoder is not available.

    The floor is what lets a question about something never said come back
    empty; without it every memory is "most similar" to something.
    """
    q = encode([query])
    if q is None:
        return None
    vecs = vectors(rows)
    if vecs is None:
        return None
    scored = [(cosine(q[0], vecs[r["id"]]), r["id"]) for r in rows]
    scored.sort(key=lambda p: -p[0])
    return [i for s, i in scored if s >= min_similarity]


def fuse(lexical: list[str], semantic: list[str], *, trusted: set[str]) -> list[str]:
    """Reciprocal-rank fusion of the meaning order with the part of the lexical
    order worth trusting, then every other lexical hit after them.

    Two choices, both measured on dev data (40 memories, 38 paraphrased
    questions; lexical alone 8/38 top-1, meaning alone 31/38):

    RANKS, NOT SCORES. The lexical order is a tuple — topic match, coverage,
    recency, specificity — not a number. A weighted sum has to invent weights
    that flatten it, fixed against whatever data was at hand, which is how the
    topic lexicon came to be overfit. RRF needs no calibration: a memory both
    orderings like beats one only either likes, and a tie goes to the lexical
    order, which carries the recency and specificity rules.

    ONLY `trusted` LEXICAL HITS ARE FUSED. Those are the topic matches, a slot
    judgement made on purpose. A bare shared stem is not: "work" in "charity
    work" retrieved the nurse's job and a brother's employer, and being found
    by both orderings let them outrank the volunteering memory the encoder had
    right. Fusing every stem hit gave 26/38; fusing topic hits only gave 31/38,
    and kept the 12/12 the lexical order scores on keyword questions where
    meaning alone drops to 10/12. Untrusted hits are still returned, after the
    fused ones, so this never loses a result the lexical search would have had.
    """
    first = [i for i in lexical if i in trusted]
    score: dict[str, float] = {}
    for ranking in (first, semantic):
        for place, ident in enumerate(ranking, 1):
            score[ident] = score.get(ident, 0.0) + 1.0 / (RRF_K + place)
    lex = {ident: place for place, ident in enumerate(first)}
    sem = {ident: place for place, ident in enumerate(semantic)}
    fused = sorted(score, key=lambda i: (-round(score[i], 12),
                                         lex.get(i, len(first)), sem.get(i, len(semantic))))
    placed = set(fused)
    return fused + [i for i in lexical if i not in placed]


class _Sims:
    """Similarity by id, worked out when asked: a write's wording lets only a few
    memories through to the comparison, and the rest never need a cosine."""

    def __init__(self, vec: array, vecs: dict[str, array]):
        self._vec, self._vecs, self._done = vec, vecs, {}

    def __getitem__(self, key: str) -> float:
        if key not in self._done:
            self._done[key] = cosine(self._vec, self._vecs[key])
        return self._done[key]


def compare(text: str, candidates: list[dict]) -> tuple[array, _Sims] | None:
    """`text`'s vector, and its similarity to each candidate. None when the
    encoder is not available."""
    q = encode([text])
    if q is None:
        return None
    vecs = vectors(candidates)
    if vecs is None:
        return None
    return q[0], _Sims(q[0], vecs)
