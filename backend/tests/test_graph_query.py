import unittest
from unittest import mock

from app import db
from app.llm import Answer, AnswerSegment
from app.rag import graph_query, query
from app.rag.graph_query import QuestionAnalysis
from tests.test_indexing import TempDbTest, fake_embed


class GraphFixture(TempDbTest):
    """Builds graph rows directly: each document has one chunk, and each assertion cites its own span."""

    def setUp(self) -> None:
        super().setUp()
        self.entities: dict[str, int] = {}
        self.docs: dict[str, int] = {}

    def doc(self, doc_id: str, status: str = "saved") -> str:
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, original_filename, extension, parser_name, parser_version, raw_path, "
                "sha256, status, uploaded_at) VALUES (?, ?, '.md', 'test', '1', '', '', ?, '2026-10-27')",
                (doc_id, f"{doc_id}.md", status),
            )
            cur = conn.execute(
                "INSERT INTO chunks (doc_id, page_number, page_title, heading_path, text) VALUES (?, 1, 'Sheet 1', '', 'x')",
                (doc_id,),
            )
            self.docs[doc_id] = cur.lastrowid
        return doc_id

    def entity(self, name: str, type_: str = "Event") -> int:
        if name not in self.entities:
            with db.connect() as conn:
                cur = conn.execute("INSERT INTO entities (canonical_name, type) VALUES (?, ?)", (name, type_))
                self.entities[name] = cur.lastrowid
        return self.entities[name]

    def link(self, src: str, dst: str, doc_id: str, *, type_: str = "causes", polarity: str = "positive",
             explicit: bool = True) -> int:
        """Adds one assertion with its own span. Returns the assertion id."""
        src_id, dst_id = self.entity(src), self.entity(dst)
        with db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO spans (chunk_id, doc_id, page_number, kind, start, end, text, llm_text, heading_path) "
                "VALUES (?, ?, 1, 'sentence', 0, 1, ?, ?, '')",
                (self.docs[doc_id], doc_id, f"{src} to {dst}", f"{src} to {dst} in {doc_id}."),
            )
            span_id = cur.lastrowid
            cur = conn.execute(
                "INSERT INTO relation_assertions (src_entity, dst_entity, type, predicate, doc_id, chunk_id, polarity, "
                "explicit, extractor_model, types_version) VALUES (?, ?, ?, 'caused', ?, ?, ?, ?, 'test', 1)",
                (src_id, dst_id, type_, doc_id, self.docs[doc_id], polarity, int(explicit)),
            )
            conn.execute("INSERT INTO assertion_spans (assertion_id, span_id) VALUES (?, ?)", (cur.lastrowid, span_id))
            return cur.lastrowid

    def build(self, name: str) -> graph_query.GraphContext | None:
        with db.connect() as conn:
            ids = graph_query._exact_targets(conn, name)
            return graph_query.build(conn, "causes_of", ids)

    def chains(self, ctx: graph_query.GraphContext) -> list[list[str]]:
        return [[ctx.names[n] for n in [p.links[-1].src] + [l.dst for l in reversed(p.links)]] for p in ctx.paths]


class TraversalTest(GraphFixture):
    def test_full_chain_survives_against_three_one_hop_causes(self) -> None:
        # C has three direct causes, and one of them has a cause of its own (A -> B -> C).
        # The two-hop chain is as strong as the direct causes, so the longer path wins the tie.
        for doc in ("d1", "d2", "d3", "d4"):
            self.doc(doc)
        self.link("X1", "C", "d1")
        self.link("X2", "C", "d2")
        self.link("X3", "C", "d3")
        self.link("B", "C", "d4")
        self.link("A", "B", "d1")
        ctx = self.build("C")
        self.assertIn(["A", "B", "C"], self.chains(ctx))
        self.assertEqual(self.chains(ctx)[0], ["A", "B", "C"])

    def test_weaker_chain_ranks_below_stronger_direct_causes(self) -> None:
        self.doc("d1")
        self.doc("d2")
        self.link("X1", "C", "d1")
        self.link("X2", "C", "d1")
        self.link("X3", "C", "d1")
        self.link("B", "C", "d2", explicit=False)
        self.link("A", "B", "d2", explicit=False)
        ctx = self.build("C")
        self.assertNotIn(["A", "B", "C"], self.chains(ctx))

    def test_prefixes_are_not_returned(self) -> None:
        self.doc("d1")
        self.link("A", "B", "d1")
        self.link("B", "C", "d1")
        ctx = self.build("C")
        self.assertEqual(self.chains(ctx), [["A", "B", "C"]])

    def test_depth_limit(self) -> None:
        self.doc("d1")
        for src, dst in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            self.link(src, dst, "d1")
        ctx = self.build("E")
        self.assertEqual(max(len(p.links) for p in ctx.paths), graph_query.GRAPH_MAX_DEPTH)

    def test_cycles_stop(self) -> None:
        self.doc("d1")
        self.link("A", "B", "d1")
        self.link("B", "A", "d1")
        self.link("B", "C", "d1")
        ctx = self.build("C")
        self.assertEqual(self.chains(ctx), [["A", "B", "C"]])

    def test_only_causal_positive_saved_links_are_followed(self) -> None:
        self.doc("d1")
        self.doc("unsaved", status="parsed")
        self.link("A", "C", "d1", type_="correlated_with")
        self.link("B", "C", "d1", polarity="negative")
        self.link("D", "C", "unsaved")
        ctx = self.build("C")
        self.assertEqual(ctx.paths, [])

    def test_negative_link_is_reported_as_conflict(self) -> None:
        self.doc("d1")
        self.doc("d2")
        self.link("B", "C", "d1")
        self.link("A", "B", "d1")
        self.link("B", "C", "d2", polarity="negative")
        ctx = self.build("C")
        self.assertEqual(len(ctx.conflicts), 1)
        self.assertIn("Disagreement", ctx.rules())

    def test_exact_matches_seed_every_entity(self) -> None:
        self.doc("d1")
        self.link("flood", "C", "d1")
        with db.connect() as conn:
            other = conn.execute("INSERT INTO entities (canonical_name, type) VALUES ('Flood', 'Event')").lastrowid
            conn.execute("INSERT INTO relation_assertions (src_entity, dst_entity, type, predicate, doc_id, chunk_id, polarity, explicit, extractor_model, types_version) VALUES (?, ?, 'causes', 'x', 'd1', ?, 'positive', 1, 'test', 1)", (other, self.entities["C"], self.docs["d1"]))
        ctx = self.build("C")
        self.assertEqual(len(ctx.paths), 2)

    def test_other_questions_skip_the_graph(self) -> None:
        self.doc("d1")
        self.link("A", "C", "d1")
        with db.connect() as conn:
            self.assertIsNone(graph_query.build(conn, "other", [self.entities["C"]]))

    def test_rules_state_the_chain(self) -> None:
        self.doc("d1")
        self.link("A", "B", "d1")
        self.link("B", "C", "d1")
        rules = self.build("C").rules()
        self.assertIn("A → B → C", rules)


class AskTest(GraphFixture):
    def test_causes_question_passes_chain_spans_and_debug(self) -> None:
        self.doc("d1")
        self.doc("d2")
        self.link("A", "B", "d1")
        self.link("B", "C", "d2")
        analysis = QuestionAnalysis(kind="causes_of", entity="C")
        seen: dict[str, object] = {}

        def fake_answer(question: str, passages: str, rules: str = "") -> tuple[Answer, str]:
            seen["passages"], seen["rules"] = passages, rules
            spans = [int(line[1:].split("]")[0]) for line in passages.splitlines() if line.startswith("[")]
            return Answer(found=True, answer_segments=[AnswerSegment(text="A → B → C", span_ids=spans)]), "{}"

        with mock.patch("app.rag.graph_query.generate_json", return_value=analysis), \
             mock.patch("app.rag.query._keyword_search", return_value=[]), \
             mock.patch("app.rag.query._vector_search", return_value=[]), \
             mock.patch("app.rag.query.answer_with_citations", side_effect=fake_answer):
            result = query.ask("What caused C?", debug=True)
        self.assertTrue(result["found"])
        self.assertIn("A to B in d1.", seen["passages"])
        self.assertIn("B to C in d2.", seen["passages"])
        self.assertIn("A → B → C", seen["rules"])
        self.assertEqual(len(result["citations"]), 2)
        graph = result["debug"]["graph"]
        self.assertEqual(graph["target"]["canonical_name"], "C")
        self.assertEqual(graph["paths"][0]["chain"], "A → B → C")
        self.assertEqual({a["src_name"] for a in graph["assertions"]}, {"A", "B"})


if __name__ == "__main__":
    unittest.main()


class ReviewTest(GraphFixture):
    def test_other_wordings_grouped_by_predicate_and_document(self) -> None:
        from app.rag import review
        self.doc("d1")
        self.doc("d2")
        self.link("A", "B", "d1", type_="other")
        self.link("C", "D", "d2", type_="other")
        self.link("E", "F", "d1", type_="other")
        with db.connect() as conn:
            conn.execute("UPDATE relation_assertions SET predicate = 'Reduces' WHERE dst_entity = ?", (self.entities["B"],))
            wordings = review.other_wordings(conn, min_docs=1)
            self.assertEqual(wordings[0]["docs"], 2)
            self.assertEqual(review.other_wordings(conn, min_docs=2)[0]["docs"], 2)

    def test_two_way_causal_pairs_reported_once(self) -> None:
        from app.rag import review
        self.doc("d1")
        self.doc("d2")
        self.link("A", "B", "d1")
        self.link("B", "A", "d2")
        self.link("C", "D", "d1")
        with db.connect() as conn:
            pairs = review.two_way_pairs(conn)
        self.assertEqual([(p["a"], p["b"]) for p in pairs], [("A", "B")])
        self.assertEqual(sorted(pairs[0]["docs"]), ["d1.md", "d2.md"])
