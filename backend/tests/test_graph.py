import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree

import httpx

from app import db, llm
from app.rag import graph_export, jobs, worker
from app.routers import documents
from app.rag.graph_extract import (
    Chunk,
    ExtractedEntity,
    ExtractedGraph,
    ExtractedRelation,
    Vocabulary,
    extract_document,
    normalize,
    validate_chunk,
)
from tests.test_indexing import TempDbTest, fake_embed

VOCAB = Vocabulary(
    version=1,
    active={"causes": "src directly causes dst", "contributes_to": "one cause among several"},
    proposed={},
    predicates={},
)
CHUNK = Chunk(
    id=1,
    heading_path="Roads",
    spans={41: "The road closure resulted from the flooding.", 42: "The flooding was not caused by the storm."},
    previous_text="",
)


def entity(name: str, span_ids: list[int], type_: str = "event") -> ExtractedEntity:
    return ExtractedEntity(name=name, type=type_, span_ids=span_ids)


def relation(src: str, dst: str, span_ids: list[int], **overrides: object) -> ExtractedRelation:
    fields: dict[str, object] = {
        "type": "causes", "predicate": "caused", "polarity": "positive", "explicit": True,
        "proposed_type": None,
    }
    fields.update(overrides)
    return ExtractedRelation(src=src, dst=dst, span_ids=span_ids, **fields)


class NormalizeTest(unittest.TestCase):
    def test_normalizes_case_space_and_punctuation(self) -> None:
        self.assertEqual(normalize("  Road-Closure,  A4 "), "road closure a4")


class ValidateChunkTest(unittest.TestCase):
    def test_unknown_span_ids_are_dropped(self) -> None:
        raw = ExtractedGraph(entities=[entity("flooding", [41, 999])], relations=[])
        graph = validate_chunk(raw, CHUNK, VOCAB)
        self.assertEqual(graph.entities["flooding"][2], {41})

    def test_entity_without_span_is_dropped(self) -> None:
        raw = ExtractedGraph(entities=[entity("storm", [])], relations=[])
        self.assertEqual(validate_chunk(raw, CHUNK, VOCAB).entities, {})

    def test_relation_needs_a_valid_span(self) -> None:
        raw = ExtractedGraph(
            entities=[entity("flooding", [42]), entity("road closure", [41])],
            relations=[relation("flooding", "road closure", [500])],
        )
        self.assertEqual(validate_chunk(raw, CHUNK, VOCAB).assertions, [])

    def test_relation_with_unknown_entity_is_dropped(self) -> None:
        raw = ExtractedGraph(
            entities=[entity("road closure", [41])],
            relations=[relation("ghost", "road closure", [41])],
        )
        self.assertEqual(validate_chunk(raw, CHUNK, VOCAB).assertions, [])

    def test_relation_spans_count_as_entity_mentions(self) -> None:
        # The model often leaves entity span_ids empty and cites only the relation.
        raw = ExtractedGraph(
            entities=[entity("flooding", []), entity("road closure", [])],
            relations=[relation("flooding", "road closure", [41])],
        )
        graph = validate_chunk(raw, CHUNK, VOCAB)
        self.assertEqual(graph.entities["flooding"][2], {41})
        self.assertEqual(graph.entities["road closure"][2], {41})
        self.assertEqual(len(graph.assertions), 1)

    def test_unknown_type_becomes_other_and_is_proposed(self) -> None:
        raw = ExtractedGraph(
            entities=[entity("flooding", [42]), entity("road closure", [41])],
            relations=[relation("flooding", "road closure", [41], type="reduces", proposed_type="Reduces")],
        )
        (assertion,) = validate_chunk(raw, CHUNK, VOCAB).assertions
        self.assertEqual((assertion.type, assertion.proposed), ("other", "reduces"))

    def test_active_type_is_not_swapped_by_passive_wording(self) -> None:
        # "The road closure resulted from the flooding": the model gives src = flooding, dst = road closure.
        # predicate_map says swap for this wording, but an active type already has the right direction.
        vocab = Vocabulary(
            1, VOCAB.active, {}, {"resulted from": ("causes", True)},
        )
        raw = ExtractedGraph(
            entities=[entity("flooding", [42]), entity("road closure", [41])],
            relations=[relation("flooding", "road closure", [41], predicate="resulted from")],
        )
        (assertion,) = validate_chunk(raw, CHUNK, vocab).assertions
        self.assertEqual((assertion.src, assertion.dst, assertion.type), ("flooding", "road closure", "causes"))

    def test_predicate_map_swaps_when_type_is_other(self) -> None:
        vocab = Vocabulary(1, VOCAB.active, {}, {"resulted from": ("causes", True)})
        raw = ExtractedGraph(
            entities=[entity("flooding", [42]), entity("road closure", [41])],
            relations=[relation("road closure", "flooding", [41], type="other", predicate="Resulted From")],
        )
        (assertion,) = validate_chunk(raw, CHUNK, vocab).assertions
        self.assertEqual((assertion.src, assertion.dst, assertion.type), ("flooding", "road closure", "causes"))
        self.assertIsNone(assertion.proposed)

    def test_inverse_proposed_type_swaps_and_is_stored_as_causes(self) -> None:
        vocab = Vocabulary(1, VOCAB.active, {"caused_by": "causes"}, {})
        raw = ExtractedGraph(
            entities=[entity("flooding", [42]), entity("road closure", [41])],
            relations=[relation("road closure", "flooding", [41], type="other", proposed_type="caused_by")],
        )
        (assertion,) = validate_chunk(raw, CHUNK, vocab).assertions
        self.assertEqual((assertion.src, assertion.dst, assertion.type), ("flooding", "road closure", "causes"))


class FakeOllama:
    """Replies to /api/chat with the given contents, in order."""

    def __init__(self, *contents: str) -> None:
        self.replies = [
            httpx.Response(200, json={"message": {"role": "assistant", "content": c}}, request=httpx.Request("POST", "http://ollama"))
            for c in contents
        ]

    def __call__(self, url: str, json: dict, timeout: float) -> httpx.Response:
        return self.replies.pop(0)


@mock.patch("app.llm.time.sleep")
class GenerateJsonTest(unittest.TestCase):
    def test_sends_schema_without_streaming(self, _sleep: mock.Mock) -> None:
        reply = json.dumps({"entities": [], "relations": []})
        with mock.patch("app.llm.httpx.post", side_effect=FakeOllama(reply)) as post:
            llm.generate_json("prompt", ExtractedGraph)
        body = post.call_args.kwargs["json"]
        self.assertFalse(body["stream"])
        self.assertTrue(body["think"])
        self.assertEqual(body["options"], {"temperature": 0})
        self.assertIn("entities", body["format"]["properties"])

    def test_invalid_reply_is_retried(self, sleep: mock.Mock) -> None:
        good = json.dumps({"entities": [], "relations": []})
        with mock.patch("app.llm.httpx.post", side_effect=FakeOllama("not json", good)) as post:
            with self.assertLogs("app.llm", "WARNING"):
                result = llm.generate_json("prompt", ExtractedGraph)
        self.assertEqual(result, ExtractedGraph(entities=[], relations=[]))
        self.assertEqual(post.call_count, 2)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2])

    def test_gives_up_after_retries_on_invalid_reply(self, _sleep: mock.Mock) -> None:
        with mock.patch("app.llm.httpx.post", side_effect=FakeOllama("x", "x", "x", "x")) as post:
            with self.assertLogs("app.llm", "WARNING"), self.assertRaises(llm.InvalidOutput):
                llm.generate_json("prompt", ExtractedGraph)
        self.assertEqual(post.call_count, 4)


class GraphTestCase(TempDbTest):
    """A saved, indexed document with two spans; extraction is patched per test."""

    def setUp(self) -> None:
        super().setUp()
        self.doc_id = self.save_doc()
        pages = [{"number": 1, "title": "Sheet 1", "markdown": "# Roads\n\nThe road closure resulted from the flooding.\n\nThe flooding was not caused by the storm.\n"}]
        documents._write_pages(f"markdown/{self.doc_id}/final.json", pages)
        with mock.patch("app.rag.indexer.embed", side_effect=fake_embed):
            jobs.enqueue([self.doc_id])
            worker.run_until_idle()

    def span_id(self, text_fragment: str) -> int:
        with db.connect() as conn:
            row = conn.execute("SELECT id FROM spans WHERE llm_text LIKE ? ORDER BY id", (f"%{text_fragment}%",)).fetchone()
        return row["id"]

    def extract_with(self, graph: ExtractedGraph) -> None:
        with mock.patch("app.rag.graph_extract.generate_json", return_value=graph):
            jobs.enqueue_graph([self.doc_id])
            worker.run_until_idle()

    def assertions(self) -> list[dict]:
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT a.type, a.polarity, s.canonical_name AS src, d.canonical_name AS dst "
                "FROM relation_assertions a "
                "JOIN entities s ON s.id = a.src_entity JOIN entities d ON d.id = a.dst_entity"
            ).fetchall()
        return [{"type": r["type"], "polarity": r["polarity"], "src": r["src"], "dst": r["dst"]} for r in rows]


class StageTest(GraphTestCase):
    def test_embed_queues_extract_and_extract_finishes(self) -> None:
        status = jobs.status_of(self.doc_id)
        self.assertEqual((status["stage"], status["status"]), ("extract", "done"))
        self.assertIsNotNone(status["graph_indexed_at"])

    def test_graph_rows_cite_spans_and_keep_direction(self) -> None:
        graph = ExtractedGraph(
            entities=[entity("road closure", [self.span_id("road closure")]), entity("flooding", [self.span_id("flooding")])],
            relations=[relation("flooding", "road closure", [self.span_id("resulted from")], predicate="resulted from")],
        )
        self.extract_with(graph)
        self.assertEqual(self.assertions(), [{"type": "causes", "polarity": "positive", "src": "flooding", "dst": "road closure"}])
        with db.connect() as conn:
            cited = conn.execute(
                "SELECT s.llm_text FROM assertion_spans asp JOIN spans s ON s.id = asp.span_id"
            ).fetchall()
        self.assertEqual([r["llm_text"] for r in cited], ["The road closure resulted from the flooding."])

    def test_reextract_replaces_rows(self) -> None:
        graph = ExtractedGraph(
            entities=[entity("road closure", [self.span_id("road closure")]), entity("flooding", [self.span_id("flooding")])],
            relations=[relation("flooding", "road closure", [self.span_id("resulted from")])],
        )
        self.extract_with(graph)
        self.extract_with(graph)
        self.assertEqual(len(self.assertions()), 1)
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) AS n FROM entities").fetchone()["n"], 2)

    def test_new_proposed_type_is_recorded_and_bumps_version(self) -> None:
        graph = ExtractedGraph(
            entities=[entity("flooding", [self.span_id("flooding")]), entity("road closure", [self.span_id("road closure")])],
            relations=[relation("flooding", "road closure", [self.span_id("resulted from")], type="other", proposed_type="reduces")],
        )
        self.extract_with(graph)
        with db.connect() as conn:
            proposed = conn.execute("SELECT name, status, class FROM relation_types WHERE status = 'proposed'").fetchall()
            version = conn.execute("SELECT value FROM index_meta WHERE key = 'relation_types_version'").fetchone()["value"]
        self.assertEqual([(r["name"], r["status"], r["class"]) for r in proposed], [("reduces", "proposed", "other")])
        self.assertEqual(version, "2")
        self.assertEqual(self.assertions()[0]["type"], "other")

    def test_stale_extract_is_discarded(self) -> None:
        self.extract_with(ExtractedGraph(entities=[], relations=[]))
        jobs.enqueue_graph([self.doc_id])
        stale = jobs.claim()
        jobs.enqueue([self.doc_id])  # saved again while the extract ran
        graph = ExtractedGraph(
            entities=[entity("flooding", [self.span_id("flooding")])],
            relations=[],
        )
        with mock.patch("app.rag.graph_extract.generate_json", return_value=graph):
            self.assertFalse(extract_document(self.doc_id, stale["generation"]))
        self.assertEqual(self.assertions(), [])
        self.assertEqual(jobs.status_of(self.doc_id)["status"], "pending")

    def test_failed_extract_keeps_index(self) -> None:
        with mock.patch("app.rag.graph_extract.generate_json", side_effect=httpx.ConnectError("refused")):
            jobs.enqueue_graph([self.doc_id])
            with self.assertLogs("app.rag.worker", "ERROR"):
                worker.run_until_idle()
        status = jobs.status_of(self.doc_id)
        self.assertEqual((status["stage"], status["status"]), ("extract", "failed"))
        self.assertIn("refused", status["error"])
        with db.connect() as conn:
            self.assertGreater(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()[0], 0)


def _export(argv: list[str]) -> int:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return graph_export.main(argv)


class ExportTest(GraphTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.extract_with(
            ExtractedGraph(
                entities=[
                    entity("road closure", [self.span_id("road closure")]),
                    entity("flooding", [self.span_id("flooding")]),
                    entity("storm", [self.span_id("storm")]),
                ],
                relations=[
                    relation("flooding", "road closure", [self.span_id("resulted from")], predicate="resulted from"),
                    relation("storm", "flooding", [self.span_id("not caused")], polarity="negative", predicate="was not caused by"),
                ],
            )
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)

    def test_expand_respects_depth(self) -> None:
        with db.connect() as conn:
            seeds = graph_export.find_entities(conn, "road closure")
            one_hop = graph_export.expand(conn, seeds, 1)
            two_hops = graph_export.expand(conn, seeds, 2)
        self.assertEqual(len(one_hop), 2)
        self.assertEqual(len(two_hops), 3)

    def test_dot_export(self) -> None:
        out = self.out / "graph.dot"
        self.assertEqual(_export(["--entity", "road closure", "--format", "dot", "--out", str(out)]), 0)
        text = out.read_text()
        self.assertTrue(text.startswith("digraph G {"))
        self.assertIn('label="causes"', text)
        self.assertIn("was not caused by", text)
        self.assertIn("style=solid", text)
        self.assertNotIn("\\\\n", text)  # a doubled backslash would print as a literal \n

    def test_graphml_export(self) -> None:
        out = self.out / "graph.graphml"
        _export(["--all", "--format", "graphml", "--out", str(out)])
        root = ElementTree.parse(out).getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        self.assertEqual(len(root.findall(".//g:node", ns)), 3)
        edges = root.findall(".//g:edge", ns)
        self.assertEqual(len(edges), 2)
        classes = {d.text for e in edges for d in e.findall("g:data", ns) if d.get("key") == "class"}
        self.assertEqual(classes, {"causal"})

    def test_html_export_is_self_contained(self) -> None:
        out = self.out / "graph.html"
        _export(["--entity", "flooding", "--depth", "1", "--out", str(out)])
        text = out.read_text()
        self.assertIn("vis-network", text)
        self.assertNotIn('src="lib/', text)
        self.assertNotIn('href="lib/', text)
        self.assertNotIn("cdn.jsdelivr.net", text)

    def test_all_mode_includes_entities_without_assertions(self) -> None:
        with db.connect() as conn:
            conn.execute("INSERT INTO entities (canonical_name, type) VALUES ('lonely', 'place')")
        out = self.out / "all.graphml"
        _export(["--all", "--format", "graphml", "--out", str(out)])
        root = ElementTree.parse(out).getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        self.assertEqual(len(root.findall(".//g:node", ns)), 4)
        self.assertEqual(len(root.findall(".//g:edge", ns)), 2)

    def test_connected_only_drops_entities_without_assertions(self) -> None:
        with db.connect() as conn:
            conn.execute("INSERT INTO entities (canonical_name, type) VALUES ('lonely', 'place')")
        out = self.out / "connected.graphml"
        _export(["--all", "--connected-only", "--format", "graphml", "--out", str(out)])
        root = ElementTree.parse(out).getroot()
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        names = {d.text for d in root.findall(".//g:node/g:data", ns) if d.get("key") == "name"}
        self.assertEqual(names, {"road closure", "flooding", "storm"})

    def test_unknown_entity_fails(self) -> None:
        self.assertEqual(_export(["--entity", "nothing here", "--out", str(self.out / "x.html")]), 1)


if __name__ == "__main__":
    unittest.main()
