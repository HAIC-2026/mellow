from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app.db import connect
from app.rag.graph_export import CAUSAL_CLASS, Edge, Graph, expand_bounded, load_graph
from app.rag.names import normalize

router = APIRouter(prefix="/api")


@router.get("/graph")
def graph_around(
    entity: int,
    depth: int = Query(default=2, ge=0, le=4),
    max_nodes: int = Query(default=150, ge=1, le=1000),
) -> dict[str, Any]:
    """Entities within depth hops of one entity, in both directions. truncated: max_nodes was reached."""
    with connect() as conn:
        if conn.execute("SELECT 1 FROM entities WHERE id = ?", (entity,)).fetchone() is None:
            raise HTTPException(404, "entity not found")
        ids, truncated = expand_bounded(conn, {entity}, depth, max_nodes)
        graph = load_graph(conn, entity_ids=ids)
    return _serialize(graph, truncated)


@router.get("/graph/entities")
def find_entities(q: str = Query(default="", max_length=200)) -> list[dict[str, Any]]:
    """Entities whose normalized name or alias contains the text. For the search box."""
    key = normalize(q)
    if not key:
        return []
    with connect() as conn:
        mentions = {r["entity_id"]: r["n"] for r in conn.execute(
            "SELECT entity_id, COUNT(*) AS n FROM entity_mentions GROUP BY entity_id"
        )}
        aliases: dict[int, list[str]] = {}
        for r in conn.execute("SELECT entity_id, alias FROM entity_aliases"):
            aliases.setdefault(r["entity_id"], []).append(r["alias"])
        matches = []
        for r in conn.execute("SELECT id, canonical_name, type FROM entities"):
            names = [normalize(r["canonical_name"]), *aliases.get(r["id"], [])]
            if any(key in name for name in names):
                matches.append({
                    "id": r["id"], "name": r["canonical_name"], "type": r["type"],
                    "mentions": mentions.get(r["id"], 0),
                })
    matches.sort(key=lambda m: (-m["mentions"], m["id"]))
    return matches[:20]


def _serialize(graph: Graph, truncated: bool) -> dict[str, Any]:
    causal_keys = {(e.src, e.dst, e.type) for e in graph.edges if e.cls == CAUSAL_CLASS}
    return {
        "nodes": [
            {"id": n.id, "name": n.name, "type": n.type, "mentions": n.mentions}
            for n in graph.nodes.values()
        ],
        "edges": [_edge(e, causal_keys) for e in graph.edges],
        "truncated": truncated,
    }


def _edge(edge: Edge, causal_keys: set[tuple[int, int, str]]) -> dict[str, Any]:
    polarities = {a.polarity for a in edge.assertions}
    return {
        "src": edge.src,
        "dst": edge.dst,
        "type": edge.type,
        "class": edge.cls,
        "documents": edge.documents,
        # Positive and negative assertions on the same link: a disagreement between documents.
        "conflict": polarities == {"positive", "negative"},
        # A causal link also stated the other way round: a likely direction error.
        "review": edge.cls == CAUSAL_CLASS and (edge.dst, edge.src, edge.type) in causal_keys,
        "assertions": [
            {
                "id": a.id, "predicate": a.predicate, "polarity": a.polarity, "explicit": a.explicit,
                "doc_id": a.doc_id, "doc_name": a.doc_name, "span_ids": a.span_ids, "spans": a.spans,
            }
            for a in edge.assertions
        ],
    }
