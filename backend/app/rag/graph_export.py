"""Export the entity graph to inspect extraction.

python -m app.rag.graph_export --entity "road closure" --depth 2 --out graph.html
python -m app.rag.graph_export --doc <doc_id> --format dot --out graph.dot
python -m app.rag.graph_export --all --format graphml --out graph.graphml
python -m app.rag.graph_export --all --connected-only --out graph.html
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from app.db import connect
from app.rag.graph_extract import normalize

CAUSAL_CLASS = "causal"
EDGE_COLOUR_NEGATIVE = "#c0392b"
EDGE_COLOUR_POSITIVE = "#555555"


@dataclass
class Node:
    id: int
    name: str
    type: str
    mentions: int


@dataclass
class Assertion:
    id: int
    src: int
    dst: int
    type: str
    predicate: str
    polarity: str
    explicit: bool
    doc_name: str
    doc_id: str
    # Span detail in the shape of /api/ask citations, so the UI can open them in SourceViewer.
    spans: list[dict[str, Any]] = field(default_factory=list)

    @property
    def span_ids(self) -> list[int]:
        return [s["span_id"] for s in self.spans]

    @property
    def span_texts(self) -> list[str]:
        return [s["text"] for s in self.spans]


@dataclass
class Edge:
    src: int
    dst: int
    type: str
    cls: str
    assertions: list[Assertion]

    @property
    def documents(self) -> int:
        return len({a.doc_id for a in self.assertions})

    @property
    def all_negative(self) -> bool:
        return all(a.polarity == "negative" for a in self.assertions)


@dataclass
class Graph:
    nodes: dict[int, Node]
    edges: list[Edge]


def load_graph(
    conn: Any, entity_ids: set[int] | None = None, doc_id: str | None = None, connected_only: bool = False
) -> Graph:
    """Entities and assertions from saved documents.

    entity_ids keeps only assertions whose two ends are both in the set. doc_id keeps only the
    assertions of that document. Neither set means everything.

    Nodes: every entity in scope, including those with no assertion. connected_only keeps only
    entities that are an end of a kept assertion.
    """
    classes = {r["name"]: r["class"] for r in conn.execute("SELECT name, class FROM relation_types")}
    rows = conn.execute(
        "SELECT a.id, a.src_entity, a.dst_entity, a.type, a.predicate, a.polarity, a.explicit, "
        "a.doc_id, d.original_filename FROM relation_assertions a JOIN documents d ON d.id = a.doc_id "
        "WHERE d.status = 'saved'"
    ).fetchall()
    kept = [
        r for r in rows
        if (doc_id is None or r["doc_id"] == doc_id)
        and (entity_ids is None or (r["src_entity"] in entity_ids and r["dst_entity"] in entity_ids))
    ]
    kept_ids = {r["id"] for r in kept}

    spans_of: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if kept_ids:
        for r in conn.execute(
            "SELECT asp.assertion_id, s.id AS span_id, s.doc_id, d.original_filename, s.page_number, "
            "c.page_title, s.heading_path, s.kind, s.start, s.end, s.text "
            "FROM assertion_spans asp JOIN spans s ON s.id = asp.span_id JOIN chunks c ON c.id = s.chunk_id "
            "JOIN documents d ON d.id = s.doc_id ORDER BY asp.assertion_id, s.start"
        ):
            if r["assertion_id"] in kept_ids:
                spans_of[r["assertion_id"]].append({k: r[k] for k in r.keys() if k != "assertion_id"})

    assertions = [
        Assertion(
            id=r["id"], src=r["src_entity"], dst=r["dst_entity"], type=r["type"], predicate=r["predicate"],
            polarity=r["polarity"], explicit=bool(r["explicit"]), doc_name=r["original_filename"],
            doc_id=r["doc_id"], spans=spans_of[r["id"]],
        )
        for r in kept
    ]
    edge_ends = {a.src for a in assertions} | {a.dst for a in assertions}
    node_ids: set[int] | None  # None: every entity
    if connected_only:
        node_ids = edge_ends
    elif entity_ids is not None:
        node_ids = entity_ids | edge_ends
    elif doc_id is not None:
        node_ids = edge_ends | {r["entity_id"] for r in conn.execute(
            "SELECT DISTINCT entity_id FROM entity_mentions WHERE doc_id = ?", (doc_id,)
        )}
    else:
        node_ids = None

    nodes: dict[int, Node] = {}
    if node_ids is None or node_ids:
        where = "" if node_ids is None else f"WHERE e.id IN ({','.join('?' * len(node_ids))})"
        params = [] if node_ids is None else list(node_ids)
        for r in conn.execute(
            "SELECT e.id, e.canonical_name, e.type, COUNT(m.span_id) AS mentions FROM entities e "
            f"LEFT JOIN entity_mentions m ON m.entity_id = e.id {where} GROUP BY e.id",
            params,
        ):
            nodes[r["id"]] = Node(r["id"], r["canonical_name"], r["type"], r["mentions"])

    grouped: dict[tuple[int, int, str], list[Assertion]] = defaultdict(list)
    for a in assertions:
        grouped[(a.src, a.dst, a.type)].append(a)
    edges = [
        Edge(src, dst, type_, classes.get(type_, "other"), items)
        for (src, dst, type_), items in sorted(grouped.items())
    ]
    return Graph(nodes, edges)


def expand_bounded(conn: Any, seeds: set[int], depth: int, max_nodes: int) -> tuple[set[int], bool]:
    """Like expand, but stops adding entities at max_nodes. Returns (entities, truncated)."""
    seen = set(seeds)
    frontier = set(seeds)
    truncated = False
    for _ in range(depth):
        if not frontier or truncated:
            break
        marks = ",".join("?" * len(frontier))
        rows = conn.execute(
            "SELECT a.src_entity, a.dst_entity FROM relation_assertions a JOIN documents d ON d.id = a.doc_id "
            f"WHERE d.status = 'saved' AND (a.src_entity IN ({marks}) OR a.dst_entity IN ({marks}))",
            [*frontier, *frontier],
        ).fetchall()
        reached = {x for r in rows for x in (r[0], r[1])} - seen
        next_frontier = set()
        for node in sorted(reached):
            if len(seen) >= max_nodes:
                truncated = True
                break
            seen.add(node)
            next_frontier.add(node)
        frontier = next_frontier
    return seen, truncated


def expand(conn: Any, seeds: set[int], depth: int) -> set[int]:
    """Entities within depth hops of the seeds, following saved assertions in both directions."""
    seen = set(seeds)
    frontier = set(seeds)
    for _ in range(depth):
        if not frontier:
            break
        marks = ",".join("?" * len(frontier))
        rows = conn.execute(
            "SELECT a.src_entity, a.dst_entity FROM relation_assertions a JOIN documents d ON d.id = a.doc_id "
            f"WHERE d.status = 'saved' AND (a.src_entity IN ({marks}) OR a.dst_entity IN ({marks}))",
            [*frontier, *frontier],
        ).fetchall()
        reached = {x for r in rows for x in (r[0], r[1])}
        frontier = reached - seen
        seen |= frontier
    return seen


def find_entities(conn: Any, text: str) -> set[int]:
    """Entities whose canonical name, or a normalized alias, matches the text."""
    ids = {r["id"] for r in conn.execute("SELECT id FROM entities WHERE lower(canonical_name) = lower(?)", (text,))}
    ids |= {r["entity_id"] for r in conn.execute("SELECT entity_id FROM entity_aliases WHERE alias = ?", (normalize(text),))}
    return ids


def tooltip(edge: Edge) -> str:
    """Plain text: vis-network does not render HTML in titles."""
    lines = [f"{edge.type} ({edge.cls}), {edge.documents} document(s)"]
    for a in edge.assertions:
        support = "explicit" if a.explicit else "inferred"
        lines.append(f"- {a.predicate!r} [{a.polarity}, {support}] {a.doc_name}")
        lines.extend(f"    {text}" for text in a.span_texts)
    return "\n".join(lines)


def to_html(graph: Graph, title: str, out: Path) -> None:
    # Imported here so the other formats work without pyvis.
    from pyvis.network import Network

    net = Network(directed=True, height="100vh", width="100%", heading=title, cdn_resources="in_line")
    for node in graph.nodes.values():
        net.add_node(
            node.id, label=node.name, title=f"{node.name} ({node.type})\n{node.mentions} mention(s)",
            size=10 + 3 * node.mentions,
        )
    for edge in graph.edges:
        net.add_edge(
            edge.src, edge.dst, label=edge.type, title=tooltip(edge), arrows="to",
            dashes=edge.cls != CAUSAL_CLASS,
            color=EDGE_COLOUR_NEGATIVE if edge.all_negative else EDGE_COLOUR_POSITIVE,
        )
    net.write_html(str(out))
    # pyvis always links Bootstrap from a CDN for its select and filter menus, which are off here.
    # Removing the links keeps the page fully offline.
    text = out.read_text(encoding="utf-8")
    text = re.sub(r"<link\b[^>]*bootstrap[^>]*/?>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"<script\b[^>]*bootstrap[^>]*>\s*</script>\s*", "", text, flags=re.DOTALL)
    out.write_text(text, encoding="utf-8")


def to_dot(graph: Graph) -> str:
    def quote(text: str) -> str:
        return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'

    lines = ["digraph G {", "  rankdir=LR;"]
    for node in graph.nodes.values():
        label = f"{node.name}\n({node.type}, {node.mentions} mentions)"
        lines.append(f"  n{node.id} [label={quote(label)}];")
    for edge in graph.edges:
        style = "solid" if edge.cls == CAUSAL_CLASS else "dashed"
        colour = EDGE_COLOUR_NEGATIVE if edge.all_negative else EDGE_COLOUR_POSITIVE
        lines.append(
            f"  n{edge.src} -> n{edge.dst} [label={quote(edge.type)}, style={style}, color={quote(colour)}, "
            f"tooltip={quote(tooltip(edge))}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def to_graphml(graph: Graph) -> bytes:
    root = ElementTree.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
    keys = {
        "name": ("node", "string"), "type": ("node", "string"), "mentions": ("node", "int"),
        "edge_type": ("edge", "string"), "class": ("edge", "string"), "documents": ("edge", "int"),
        "assertions": ("edge", "string"),
    }
    for key_id, (scope, kind) in keys.items():
        ElementTree.SubElement(
            root, "key", id=key_id, **{"for": scope, "attr.name": key_id, "attr.type": kind}
        )
    xml_graph = ElementTree.SubElement(root, "graph", id="G", edgedefault="directed")

    def data(parent: ElementTree.Element, key: str, value: object) -> None:
        ElementTree.SubElement(parent, "data", key=key).text = str(value)

    for node in graph.nodes.values():
        el = ElementTree.SubElement(xml_graph, "node", id=f"n{node.id}")
        data(el, "name", node.name)
        data(el, "type", node.type)
        data(el, "mentions", node.mentions)
    for edge in graph.edges:
        el = ElementTree.SubElement(xml_graph, "edge", source=f"n{edge.src}", target=f"n{edge.dst}")
        data(el, "edge_type", edge.type)
        data(el, "class", edge.cls)
        data(el, "documents", edge.documents)
        data(el, "assertions", tooltip(edge))
    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--entity", help="canonical name or alias; the graph is drawn around it")
    scope.add_argument("--doc", help="document id; the entities and assertions of one document")
    scope.add_argument("--all", action="store_true", help="every entity and saved assertion")
    parser.add_argument("--depth", type=int, default=2, help="hops from --entity (default 2)")
    parser.add_argument(
        "--connected-only", action="store_true", help="drop entities that have no assertion"
    )
    parser.add_argument("--format", choices=["html", "dot", "graphml"], default="html")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    with connect() as conn:
        if args.entity is not None:
            seeds = find_entities(conn, args.entity)
            if not seeds:
                print(f"no entity named or aliased {args.entity!r}", file=sys.stderr)
                return 1
            graph = load_graph(conn, entity_ids=expand(conn, seeds, args.depth), connected_only=args.connected_only)
            title = f"{args.entity} (depth {args.depth})"
        elif args.doc is not None:
            graph = load_graph(conn, doc_id=args.doc, connected_only=args.connected_only)
            title = f"document {args.doc}"
        else:
            graph = load_graph(conn, connected_only=args.connected_only)
            title = "all entities"

    if args.format == "html":
        to_html(graph, title, args.out)
    elif args.format == "dot":
        args.out.write_text(to_dot(graph), encoding="utf-8")
    else:
        args.out.write_bytes(to_graphml(graph))
    print(f"wrote {args.out}: {len(graph.nodes)} entities, {len(graph.edges)} edges")
    return 0


if __name__ == "__main__":
    sys.exit(main())
