"""Entity resolution, stage extract: attach a document's names to existing entities.

Rules are in docs/graph-rag-v2.md, section 4. A merge needs identity: the same normalized name
and type, or a "same" answer from the model. Embeddings only propose candidates, so related
names such as "flood" and "flood maps" stay apart when the model says "different".
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from app.llm import embed, generate_json
from app.rag.names import normalize

log = logging.getLogger(__name__)

# Measured on the development corpus (qwen3-embedding, names only). Same-entity full names
# scored 0.87-0.97. Abbreviations (DBEDT, HI-EMA) scored 0.51-0.62 and never reach the model.
# Unrelated pairs scored 0.71-0.87, so above this threshold the model decides.
RESOLVE_MIN_COSINE = 0.75
RESOLVE_MAX_CANDIDATES = 3

PROMPT = """You decide whether two names from one document refer to the same real-world entity.

Name A: "{a_name}" (type: {a_type})
Passage for A: {a_sample}

Name B: "{b_name}" (type: {b_type})
Passage for B: {b_sample}

Answer "same" only if the two names are two ways of writing one thing: an abbreviation and its
full name, a name and its variant, or a singular and plural of one thing. Answer "different" if
one name is a part of the other (the islands of a state are not the state), a group or
collection of the other, a cause or effect of it, a map or record of it, or a related but
separate thing. A name that is more specific than the other ("heavy traffic jams" against
"traffic jams") is different in identity terms. Answer "unsure" if the passages do not show which.
"""


class SameAnswer(BaseModel):
    decision: Literal["same", "different", "unsure"]


@dataclass(frozen=True)
class Name:
    """A name extracted from this document, with a passage that states it."""

    key: str  # normalized name
    name: str
    type: str
    sample: str


@dataclass(frozen=True)
class Existing:
    """An entity that already has mentions in another document."""

    id: int
    name: str
    type: str
    mentions: int
    norms: frozenset[str]  # normalized surface forms from other documents
    sample: str
    vector: np.ndarray | None  # unit length; None if embedded with another model or not at all


@dataclass(frozen=True)
class Plan:
    """Where a name goes. existing: merge into it. Otherwise create the entity of `root`."""

    existing: Existing | None
    root: str  # key of the name that creates the entity (this name, or an earlier one it joins)


@dataclass(frozen=True)
class Resolution:
    plans: dict[str, Plan]
    vectors: dict[str, bytes]  # float32 bytes of each new name's embedding, by key


@dataclass(frozen=True)
class _Candidate:
    name: str
    type: str
    sample: str
    existing: Existing | None  # None: a name created earlier in this run
    root: str


def load_pool(conn: Any, doc_id: str, model_key: str) -> list[Existing]:
    """Entities with mentions in documents other than doc_id.

    Only other documents count. Entities that this document alone supports are rebuilt from
    scratch when it is extracted, so the names it resolves against must not include them.
    """
    counts = {r["entity_id"]: r["n"] for r in conn.execute(
        "SELECT entity_id, COUNT(*) AS n FROM entity_mentions GROUP BY entity_id"
    )}
    norms: dict[int, set[str]] = {}
    info: dict[int, Any] = {}
    rows = conn.execute(
        "SELECT m.entity_id, m.surface_form, s.llm_text, e.canonical_name, e.type, e.embedding, e.embed_model "
        "FROM entity_mentions m JOIN entities e ON e.id = m.entity_id JOIN spans s ON s.id = m.span_id "
        "WHERE m.doc_id != ? ORDER BY m.id",
        (doc_id,),
    ).fetchall()
    for r in rows:
        if r["entity_id"] not in info:
            info[r["entity_id"]] = r
        norms.setdefault(r["entity_id"], set()).add(normalize(r["surface_form"]))
    pool = []
    for entity_id, r in info.items():
        vector = None
        if r["embedding"] is not None and r["embed_model"] == model_key:
            vector = _unit(np.frombuffer(r["embedding"], dtype=np.float32))
        pool.append(Existing(
            id=entity_id,
            name=r["canonical_name"],
            type=r["type"],
            mentions=counts.get(entity_id, 0),
            norms=frozenset(norms[entity_id]),
            sample=r["llm_text"],
            vector=vector,
        ))
    return pool


def resolve_names(names: dict[str, Name], pool: list[Existing], model_key: str) -> Resolution:
    """Plan every name of one document. Calls the model for candidates; writes nothing.

    Names are resolved in order. A name that creates an entity becomes a candidate for the
    names after it, so two names in one document can merge.
    """
    if not names:
        return Resolution({}, {})
    raw = embed([n.name for n in names.values()], "RETRIEVAL_DOCUMENT")
    vectors = {key: np.asarray(v, dtype=np.float32) for key, v in zip(names, raw)}
    created: list[_Candidate] = []
    created_vectors: dict[str, np.ndarray] = {}
    plans: dict[str, Plan] = {}
    for key, name in names.items():
        vector = _unit(vectors[key])
        plans[key] = _resolve_one(name, vector, pool, created, created_vectors)
        if plans[key].existing is None and plans[key].root == key:
            created.append(_Candidate(name.name, name.type, name.sample, None, key))
            created_vectors[key] = vector
    return Resolution(plans, {key: vectors[key].tobytes() for key in names})


def _resolve_one(
    name: Name,
    vector: np.ndarray,
    pool: list[Existing],
    created: list[_Candidate],
    created_vectors: dict[str, np.ndarray],
) -> Plan:
    confirm: list[_Candidate] = []
    exact = [p for p in pool if name.key in p.norms]
    if exact:
        # Same normalized name: identity. Prefer the same type, then the most mentions, then the lowest id.
        best = max(exact, key=lambda p: (normalize(p.type) == normalize(name.type), p.mentions, -p.id))
        if normalize(best.type) == normalize(name.type):
            return Plan(best, root=name.key)
        confirm.append(_Candidate(best.name, best.type, best.sample, best, name.key))

    ranked: list[tuple[float, _Candidate]] = []
    exact_ids = {c.existing.id for c in confirm if c.existing is not None}
    for p in pool:
        if p.vector is None or p.id in exact_ids:
            continue
        score = float(p.vector @ vector)
        if score >= RESOLVE_MIN_COSINE:
            ranked.append((score, _Candidate(p.name, p.type, p.sample, p, name.key)))
    for c in created:
        score = float(created_vectors[c.root] @ vector)
        if score >= RESOLVE_MIN_COSINE:
            ranked.append((score, c))
    ranked.sort(key=lambda t: -t[0])
    confirm += [c for _, c in ranked]

    for candidate in confirm[:RESOLVE_MAX_CANDIDATES]:
        decision = _decide(name, candidate)
        log.debug("resolve %r with %r: %s", name.name, candidate.name, decision)
        if decision == "same":
            return Plan(candidate.existing, root=name.key if candidate.existing else candidate.root)
        if decision == "unsure":
            break
    return Plan(None, root=name.key)


def _decide(name: Name, candidate: _Candidate) -> str:
    return same_entity(
        name.name, name.type, name.sample, candidate.name, candidate.type, candidate.sample
    )


def same_entity(a_name: str, a_type: str, a_sample: str, b_name: str, b_type: str, b_sample: str) -> str:
    """Model answer: "same", "different" or "unsure" for two names with one passage each."""
    prompt = PROMPT.format(
        a_name=a_name, a_type=a_type, a_sample=a_sample,
        b_name=b_name, b_type=b_type, b_sample=b_sample,
    )
    # Non-thinking: the decision is short and there are many candidates per document.
    return generate_json(prompt, SameAnswer, think=False).decision


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector
