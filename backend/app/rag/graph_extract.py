"""Graph extraction, stage extract: entities and relations per chunk, validated against the spans.

Rules are in docs/graph-rag-v2.md, sections 1 and 3. Entities are one per normalized name
within a document; resolution across documents is phase 3.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel

from app.db import connect
from app.llm import embed_model_key, extract_model, generate_json
from app.rag import jobs
from app.rag.graph_resolve import Existing, Name, Resolution, load_pool, resolve_names
from app.rag.names import normalize

log = logging.getLogger(__name__)

PROMPT = """You extract entities and relations from numbered passages of one document.

Entities are the named things the passages discuss: events, places, organizations,
people, roads, measures. Use the name as the text states it. Give a short type.

A relation links two entities that the passages state. Relation types:
{types}
- Use "other" when no type above fits. Then set proposed_type to a short snake_case name
  for the relation. Otherwise set proposed_type to null.

Direction: src is always the cause or the subject; dst is the effect or the object.
Decide by the grammar, not by the word order:
- "A caused B", "A led to B", "A is due to B" -> src = A, dst = B.
- "B was caused by A", "B resulted from A", "B is due to A", "B stems from A",
  "B was driven by A" -> src = A, dst = B.
- "A did not cause B" -> src = A, dst = B, polarity = negative.
- "A was not caused by B" -> src = B, dst = A, polarity = negative.

polarity: "negative" when the passage denies the relation, otherwise "positive".
explicit: true when a passage states the relation, false when you infer it.
predicate: the words the passage uses for the relation, copied exactly.

span_ids: for an entity, the ids of the Target passages that mention it; for a relation, the ids
of the Target passages that state it. Every entity and relation needs at least one id. Cite
only Target passage ids. The Context below has no ids and must not be cited.

Context (for resolving pronouns and names; do not extract relations from it alone):
Section: {heading}
Previous passage: {previous}

Target passages:
{passages}
"""


class ExtractedEntity(BaseModel):
    name: str
    type: str
    span_ids: list[int]


class ExtractedRelation(BaseModel):
    src: str
    dst: str
    type: str
    predicate: str
    polarity: Literal["positive", "negative"]
    explicit: bool
    span_ids: list[int]
    proposed_type: str | None = None


class ExtractedGraph(BaseModel):
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


@dataclass(frozen=True)
class Chunk:
    id: int
    heading_path: str
    spans: dict[int, str]  # span id -> llm_text, in reading order
    previous_text: str


@dataclass(frozen=True)
class Vocabulary:
    version: int
    active: dict[str, str]  # active type -> description
    proposed: dict[str, str | None]  # proposed type -> inverse_of
    predicates: dict[str, tuple[str, bool]]  # normalized wording -> (type, swap)


@dataclass
class Assertion:
    src: str  # normalized entity name, after any swap
    dst: str
    type: str
    predicate: str
    polarity: str
    explicit: bool
    span_ids: list[int]
    proposed: str | None  # a new proposed type to record, when the stored type is 'other'


@dataclass
class ChunkGraph:
    chunk_id: int
    entities: dict[str, tuple[str, str, set[int]]]  # normalized name -> (name, type, span ids)
    assertions: list[Assertion]


def load_vocabulary(conn: Any) -> Vocabulary:
    rows = conn.execute("SELECT name, description, status, inverse_of FROM relation_types").fetchall()
    version = conn.execute("SELECT value FROM index_meta WHERE key = 'relation_types_version'").fetchone()
    predicates = {
        normalize(r["predicate"]): (r["type"], bool(r["swap"]))
        for r in conn.execute("SELECT predicate, type, swap FROM predicate_map")
    }
    return Vocabulary(
        version=int(version["value"]),
        active={r["name"]: r["description"] for r in rows if r["status"] == "active"},
        proposed={r["name"]: r["inverse_of"] for r in rows if r["status"] == "proposed"},
        predicates=predicates,
    )


def load_chunks(conn: Any, doc_id: str) -> list[Chunk]:
    chunk_rows = conn.execute(
        "SELECT id, heading_path FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
    ).fetchall()
    spans_by_chunk: dict[int, dict[int, str]] = {}
    for row in conn.execute("SELECT id, chunk_id, llm_text FROM spans WHERE doc_id = ? ORDER BY id", (doc_id,)):
        spans_by_chunk.setdefault(row["chunk_id"], {})[row["id"]] = row["llm_text"]
    chunks = []
    previous = ""
    for row in chunk_rows:
        spans = spans_by_chunk.get(row["id"])
        if not spans:
            continue
        chunks.append(Chunk(row["id"], row["heading_path"], spans, previous))
        previous = "\n".join(spans.values())
    return chunks


def build_prompt(chunk: Chunk, vocab: Vocabulary) -> str:
    types = "\n".join(f"- {name}: {desc}" for name, desc in vocab.active.items())
    passages = "\n".join(f"[{span_id}] {text}" for span_id, text in chunk.spans.items())
    return PROMPT.format(
        types=types,
        heading=chunk.heading_path or "(none)",
        previous=chunk.previous_text or "(none)",
        passages=passages,
    )


def validate_chunk(raw: ExtractedGraph, chunk: Chunk, vocab: Vocabulary) -> ChunkGraph:
    """Drop everything the spans do not support. Nothing without a valid span id is kept.

    A relation's spans also count as mentions of its two entities: the model often gives
    relations spans and leaves the entity lists empty. Entities left with no span are dropped.
    """
    names: dict[str, tuple[str, str]] = {}
    spans: dict[str, set[int]] = {}
    for e in raw.entities:
        key = normalize(e.name)
        if not key:
            continue
        names.setdefault(key, (e.name.strip(), e.type.strip()))
        spans.setdefault(key, set()).update(i for i in e.span_ids if i in chunk.spans)

    assertions = []
    for r in raw.relations:
        src, dst = normalize(r.src), normalize(r.dst)
        span_ids = list(dict.fromkeys(i for i in r.span_ids if i in chunk.spans))
        if src not in names or dst not in names or not span_ids:
            continue
        type_, swap, proposed = _resolve_type(r, vocab)
        if swap:
            src, dst = dst, src
        spans[src].update(span_ids)
        spans[dst].update(span_ids)
        assertions.append(Assertion(src, dst, type_, r.predicate.strip(), r.polarity, r.explicit, span_ids, proposed))

    entities = {key: (names[key][0], names[key][1], ids) for key, ids in spans.items() if ids}
    return ChunkGraph(chunk.id, entities, assertions)


def _resolve_type(r: ExtractedRelation, vocab: Vocabulary) -> tuple[str, bool, str | None]:
    """Returns (stored type, swap src and dst, proposed type to record).

    Swaps apply only when the type is not active. An active type already fixes the direction,
    so swapping it again would reverse a correct edge.
    """
    type_ = r.type.strip()
    if type_ in vocab.active:
        return type_, False, None
    proposed = _type_name(r.proposed_type) or (_type_name(type_) if type_ != "other" else None)
    if proposed in vocab.active:
        return proposed, False, None
    if proposed in vocab.proposed:
        inverse = vocab.proposed[proposed]
        if inverse is not None:
            return inverse, True, None
    mapped = vocab.predicates.get(normalize(r.predicate))
    if mapped is not None:
        return mapped[0], mapped[1], None
    return "other", False, proposed


def _type_name(name: str | None) -> str | None:
    if not name or not name.strip():
        return None
    return "_".join(name.strip().lower().split())


def extract_document(doc_id: str, generation: int) -> bool:
    """Run stage extract. Returns False without writing if the document was saved again meanwhile.

    The model calls (extraction, then resolution) run outside the write lock. The transaction
    re-checks each resolved entity, so an entity deleted meanwhile is created again, not reused.
    """
    model_key = embed_model_key()
    with connect() as conn:
        vocab = load_vocabulary(conn)
        chunks = load_chunks(conn, doc_id)
        pool = load_pool(conn, doc_id, model_key)
    chunk_by_id = {c.id: c for c in chunks}
    graphs = []
    for chunk in chunks:
        raw = generate_json(build_prompt(chunk, vocab), ExtractedGraph)
        graphs.append(validate_chunk(raw, chunk, vocab))
    names = _doc_names(graphs, chunk_by_id)
    resolution = resolve_names(names, pool, model_key)

    with connect() as conn:
        # Write lock before the generation check, as in indexer.index_document.
        conn.execute("BEGIN IMMEDIATE")
        if not jobs.is_current(conn, doc_id, generation):
            log.info("discarding stale extract run for %s (generation %s)", doc_id, generation)
            return False
        delete_graph(conn, doc_id)
        _insert_graph(conn, doc_id, graphs, names, resolution, vocab.version, extract_model(), model_key)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("UPDATE documents SET graph_indexed_at = ? WHERE id = ?", (now, doc_id))
        conn.execute(
            "UPDATE index_jobs SET status = 'done', error = NULL, updated_at = ? WHERE doc_id = ?", (now, doc_id)
        )
    return True


def _doc_names(graphs: list[ChunkGraph], chunk_by_id: dict[int, Chunk]) -> dict[str, Name]:
    """One name per normalized entity name in the document, with the first passage that states it."""
    names: dict[str, Name] = {}
    for graph in graphs:
        for key, (name, type_, span_ids) in graph.entities.items():
            if key not in names:
                names[key] = Name(key, name, type_, chunk_by_id[graph.chunk_id].spans[min(span_ids)])
    return names


def delete_graph(conn: Any, doc_id: str) -> None:
    affected = [r["entity_id"] for r in conn.execute(
        "SELECT DISTINCT entity_id FROM entity_mentions WHERE doc_id = ?", (doc_id,)
    )]
    conn.execute(
        "DELETE FROM assertion_spans WHERE assertion_id IN (SELECT id FROM relation_assertions WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM relation_assertions WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM entity_mentions WHERE doc_id = ?", (doc_id,))
    # Entities left without mentions go too. Other documents' entities keep their mentions.
    conn.execute("DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM entity_mentions)")
    refresh_aliases(conn, affected)


def refresh_aliases(conn: Any, entity_ids: Iterable[int]) -> None:
    """Aliases are the normalized surface forms of an entity's current mentions (doc section 4)."""
    ids = list(dict.fromkeys(entity_ids))
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM entity_aliases WHERE entity_id IN ({marks})", ids)
    rows = conn.execute(
        f"SELECT DISTINCT entity_id, surface_form FROM entity_mentions WHERE entity_id IN ({marks})", ids
    ).fetchall()
    conn.executemany(
        "INSERT OR IGNORE INTO entity_aliases (entity_id, alias) VALUES (?, ?)",
        [(r["entity_id"], normalize(r["surface_form"])) for r in rows if normalize(r["surface_form"])],
    )


def _insert_graph(
    conn: Any,
    doc_id: str,
    graphs: list[ChunkGraph],
    names: dict[str, Name],
    resolution: Resolution,
    version: int,
    model: str,
    model_key: str,
) -> None:
    ids: dict[str, int] = {}
    for key in names:
        plan = resolution.plans[key]
        if plan.existing is not None and _is_unchanged(conn, plan.existing):
            ids[key] = plan.existing.id
            continue
        root = plan.root if plan.existing is None else key
        if root not in ids:
            ids[root] = _insert_entity(conn, names[root], resolution.vectors[root], model_key)
        ids[key] = ids[root]

    for graph in graphs:
        for key, (name, _type, span_ids) in graph.entities.items():
            conn.executemany(
                "INSERT INTO entity_mentions (entity_id, span_id, doc_id, surface_form) VALUES (?, ?, ?, ?)",
                [(ids[key], span_id, doc_id, name) for span_id in sorted(span_ids)],
            )
        for a in graph.assertions:
            if a.proposed is not None:
                _record_proposed(conn, a.proposed)
            cur = conn.execute(
                "INSERT INTO relation_assertions (src_entity, dst_entity, type, predicate, doc_id, chunk_id, "
                "polarity, explicit, extractor_model, types_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ids[a.src], ids[a.dst], a.type, a.predicate, doc_id, graph.chunk_id,
                    a.polarity, int(a.explicit), model, version,
                ),
            )
            conn.executemany(
                "INSERT INTO assertion_spans (assertion_id, span_id) VALUES (?, ?)",
                [(cur.lastrowid, span_id) for span_id in a.span_ids],
            )
    refresh_aliases(conn, ids.values())


def _is_unchanged(conn: Any, existing: Existing) -> bool:
    # A resolved entity may have been removed, or its id reused, since resolution ran.
    row = conn.execute("SELECT canonical_name, type FROM entities WHERE id = ?", (existing.id,)).fetchone()
    return row is not None and (row["canonical_name"], row["type"]) == (existing.name, existing.type)


def _insert_entity(conn: Any, name: Name, vector: bytes, model_key: str) -> int:
    cur = conn.execute(
        "INSERT INTO entities (canonical_name, type, embedding, embed_model) VALUES (?, ?, ?, ?)",
        (name.name, name.type, vector, model_key),
    )
    return cur.lastrowid


def _record_proposed(conn: Any, name: str) -> None:
    # A new proposed type changes relation_types, so it bumps the version (docs 1.3, step 4).
    cur = conn.execute(
        "INSERT OR IGNORE INTO relation_types (name, description, class, status, inverse_of) "
        "VALUES (?, 'proposed by extraction', 'other', 'proposed', NULL)",
        (name,),
    )
    if cur.rowcount == 1:
        conn.execute(
            "UPDATE index_meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'relation_types_version'"
        )
