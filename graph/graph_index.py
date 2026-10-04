"""A graph over memories, used as an index to choose which facts go into the prompt.

Today every chat carries a block of all current facts (up to 200, ~3.2k
tokens). Here each fact is linked to the people, pets, places, things and
topic it mentions; a message seeds the graph with its own names / topic and
with its 3 most similar facts, and a personalized PageRank walk over
fact <-> node links ranks the facts (the HippoRAG idea, without any LLM). The
top k go into the block; safety facts (allergies, medication) always do.

No language model anywhere: names come from capitalisation and possessives,
topics from Primnox's own word lexicon (backend/primnox2/cognition/topics.py),
keywords from content words weighted by rarity.
"""
from __future__ import annotations

import importlib.util
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("topics", ROOT / "backend/primnox2/cognition/topics.py")
topics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(topics)

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")
CAP = re.compile(r"[A-Z][a-zA-Z'\-]+")
KIN = re.compile(r"\bmy (wife|husband|partner|boyfriend|girlfriend|fianc[eé]e?|son|daughter|kids?|children|mum|mom|"
                 r"mother|dad|father|parents|brother|sister|grandma|grandmother|grandpa|grandfather|aunt|uncle|"
                 r"cousin|niece|nephew|dog|cat|puppy|kitten|pet|boss|manager|landlord|flatmate|roommate|team|"
                 r"car|bike|flat|apartment|house|phone|laptop|gym|doctor|therapist|job|course|school)\b", re.I)
STOP = set("""a an the and or but if so of to in on at by for with from up down out over under about into than then
too very can will just not no nor only own same such that this these those there here when where why how what who
whom which while is am are was were be been being have has had having do does did doing i me my mine myself we us
our ours you your yours he him his she her hers it its they them their theirs im ive id ill dont didnt cant wont isnt
get got getting go going went gone still also now really much many lot lots some any all every each both few more
most other another again ever never always one two last next first new old good bad like just back since week weeks
month months year years day days today yesterday tomorrow time thing things way bit""".split())


def content_words(text: str) -> set[str]:
    out = set()
    for w in WORD.findall(text.lower()):
        w = w.strip("'-")
        if len(w) > 2 and w not in STOP:
            out.add(w[:-1] if w.endswith("s") and len(w) > 4 else w)   # crude plural folding
    return out


class FactGraph:
    def __init__(self, facts: list[dict], vec: dict):
        """facts: [{"id", "text"}]; vec: text -> normalised embedding (torch tensor)."""
        self.facts, self.vec = facts, vec
        self.names = self._names()
        self.links: dict[str, set[str]] = {}           # fact id -> node ids
        self.members: dict[str, set[str]] = defaultdict(set)
        for f in facts:
            nodes = self.nodes_of(f["text"])
            self.links[f["id"]] = nodes
            for n in nodes:
                self.members[n].add(f["id"])
        n_facts = max(1, len(facts))
        # a node linked to every fact carries no information: weight by rarity
        self.node_weight = {n: math.log(1 + n_facts / len(m)) for n, m in self.members.items()}

    def _names(self) -> set[str]:
        seen = set()
        for f in self.facts:
            for m in CAP.finditer(f["text"]):
                before = f["text"][:m.start()].rstrip()
                if before and before[-1] not in ".!?\"“(":
                    seen.add(m.group().split("'")[0])
        return seen - {"I", "I'm", "I've", "I'd", "I'll"}

    def nodes_of(self, text: str) -> set[str]:
        nodes = {"name:" + w.split("'")[0] for w in CAP.findall(text) if w.split("'")[0] in self.names}
        nodes |= {"kin:" + m.group(1).lower() for m in KIN.finditer(text)}
        t = topics.topic_of(text)
        if t:
            nodes.add("topic:" + t)
        nodes |= {"word:" + w for w in content_words(text)}
        return nodes

    def query_nodes(self, query: str) -> set[str]:
        nodes = {n for n in self.nodes_of(query) if n in self.members}
        t = topics.topic_of_query(query)
        if t and "topic:" + t in self.members:
            nodes.add("topic:" + t)
        return nodes

    def similar(self, query_vec, k: int) -> list[str]:
        scored = sorted(self.facts, key=lambda f: -float(self.vec[f["text"]] @ query_vec))
        return [f["id"] for f in scored[:k]]

    def rank(self, query: str, query_vec, seeds_from_similarity: int = 3, alpha: float = 0.5,
             iters: int = 20) -> list[tuple[str, float]]:
        """Personalized PageRank over the fact <-> node bipartite graph, restarting at the seeds."""
        seed_nodes = self.query_nodes(query)
        seed_facts = set(self.similar(query_vec, seeds_from_similarity))
        restart: dict[str, float] = defaultdict(float)
        for n in seed_nodes:
            restart[n] += self.node_weight.get(n, 1.0)
        for fid in seed_facts:
            restart["fact:" + fid] += 1.0
        total = sum(restart.values()) or 1.0
        restart = {k: v / total for k, v in restart.items()}
        score = dict(restart)
        for _ in range(iters):
            nxt: dict[str, float] = defaultdict(float)
            for key, mass in score.items():
                if key.startswith("fact:"):
                    nodes = self.links.get(key[5:], ())
                    w = sum(self.node_weight.get(n, 1.0) for n in nodes) or 1.0
                    for n in nodes:
                        nxt[n] += (1 - alpha) * mass * self.node_weight.get(n, 1.0) / w
                else:
                    members = self.members.get(key, ())
                    for fid in members:
                        nxt["fact:" + fid] += (1 - alpha) * mass / len(members)
            for k, v in restart.items():
                nxt[k] += alpha * v
            score = nxt
        sim = {f["id"]: float(self.vec[f["text"]] @ query_vec) for f in self.facts}
        # graph score first, similarity breaks ties and orders facts the walk never reached
        return sorted(((f["id"], score.get("fact:" + f["id"], 0.0)) for f in self.facts),
                      key=lambda x: (-x[1], -sim[x[0]]))

    def select(self, query: str, query_vec, k: int) -> list[str]:
        safety = [f["id"] for f in self.facts if topics.is_safety(f["text"])]
        ranked = [fid for fid, _ in self.rank(query, query_vec) if fid not in safety]
        return safety + ranked[:max(0, k - len(safety))]
