import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from app import db, llm
from app.rag import jobs, query, worker
from app.rag.graph_extract import ExtractedGraph
from app.rag.graph_query import QuestionAnalysis
from app.rag.graph_resolve import SameAnswer
from app.routers import documents

PAGES = [{"number": 1, "title": "Sheet 1", "markdown": "# Roads\n\nThe road closure resulted from the flooding.\n"}]


def fake_embed(texts: list[str], task_type: str) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in texts]


class TempDbTest(unittest.TestCase):
    """Each test gets its own data directory and SQLite database."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        data_dir = Path(tmp.name)
        for target, value in [
            ("app.db.DATA_DIR", data_dir),
            ("app.db.DB_PATH", data_dir / "app.db"),
            ("app.routers.documents.DATA_DIR", data_dir),
        ]:
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict("os.environ", {"OLLAMA_EMBED_MODEL": "qwen3-embedding", "OLLAMA_EMBED_DIM": "3"})
        env.start()
        self.addCleanup(env.stop)
        # Stage extract calls the extraction model, the resolver embeds names and asks for
        # same/different answers. Tests that need graph output patch these again.
        for target, kwargs in [
            ("app.rag.graph_extract.generate_json", {"return_value": ExtractedGraph(entities=[], relations=[])}),
            ("app.rag.graph_resolve.embed", {"side_effect": fake_embed}),
            ("app.rag.graph_resolve.generate_json", {"return_value": SameAnswer(decision="different")}),
            ("app.rag.graph_query.embed", {"side_effect": fake_embed}),
            ("app.rag.graph_query.generate_json", {"return_value": QuestionAnalysis(kind="other", entity="")}),
        ]:
            patcher = mock.patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        db.init_db()

    def save_doc(self, doc_id: str = "doc1") -> str:
        rel = f"markdown/{doc_id}/final.json"
        documents._write_pages(rel, PAGES)
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO documents (id, original_filename, extension, parser_name, parser_version, "
                "raw_path, final_md_path, sha256, status, uploaded_at) "
                "VALUES (?, 'roads.md', '.md', 'test', '1', '', ?, '', 'saved', '2026-10-27')",
                (doc_id, rel),
            )
        return doc_id

    def chunk_count(self, doc_id: str) -> int:
        with db.connect() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()["n"]


@mock.patch("app.rag.indexer.embed", side_effect=fake_embed)
class WorkerTest(TempDbTest):
    def test_job_indexes_document(self, _embed: mock.Mock) -> None:
        doc_id = self.save_doc()
        jobs.enqueue([doc_id])
        worker.run_until_idle()
        status = jobs.status_of(doc_id)
        self.assertEqual((status["stage"], status["status"]), ("extract", "done"))
        self.assertIsNotNone(status["indexed_at"])
        self.assertIsNotNone(status["graph_indexed_at"])
        with db.connect() as conn:
            models = {r["embed_model"] for r in conn.execute("SELECT embed_model FROM chunks")}
        self.assertEqual(models, {"qwen3-embedding@3"})

    def test_stale_run_is_discarded(self, _embed: mock.Mock) -> None:
        doc_id = self.save_doc()
        jobs.enqueue([doc_id])
        stale = jobs.claim()
        jobs.enqueue([doc_id])  # saved again while the first run is in progress
        worker.run_job(stale)
        self.assertEqual(self.chunk_count(doc_id), 0)
        self.assertEqual(jobs.status_of(doc_id)["status"], "pending")
        worker.run_until_idle()
        self.assertEqual(jobs.status_of(doc_id)["status"], "done")
        self.assertGreater(self.chunk_count(doc_id), 0)

    def test_failure_keeps_old_index(self, embed: mock.Mock) -> None:
        doc_id = self.save_doc()
        jobs.enqueue([doc_id])
        worker.run_until_idle()
        before = self.chunk_count(doc_id)
        embed.side_effect = httpx.ConnectError("connection refused")
        jobs.enqueue([doc_id])
        with self.assertLogs("app.rag.worker", "ERROR"):
            worker.run_until_idle()
        status = jobs.status_of(doc_id)
        self.assertEqual(status["status"], "failed")
        self.assertIn("connection refused", status["error"])
        self.assertEqual(self.chunk_count(doc_id), before)

    def test_document_list_carries_index_status(self, _embed: mock.Mock) -> None:
        doc_id = self.save_doc()
        jobs.enqueue([doc_id])
        (row,) = documents.list_documents()
        self.assertEqual(row["index_status"]["status"], "pending")
        worker.run_until_idle()
        (row,) = documents.list_documents()
        self.assertEqual(row["index_status"]["status"], "done")

    def test_job_is_claimed_once(self, _embed: mock.Mock) -> None:
        jobs.enqueue([self.save_doc()])
        self.assertIsNotNone(jobs.claim())
        self.assertIsNone(jobs.claim())

    def test_restart_resumes_running_jobs(self, _embed: mock.Mock) -> None:
        doc_id = self.save_doc()
        jobs.enqueue([doc_id])
        jobs.claim()  # the process stops here, leaving the job running
        jobs.reset_running()
        worker.run_until_idle()
        self.assertEqual(jobs.status_of(doc_id)["status"], "done")


class VectorSearchTest(TempDbTest):
    def index(self, doc_id: str) -> None:
        with mock.patch("app.rag.indexer.embed", side_effect=fake_embed):
            jobs.enqueue([doc_id])
            worker.run_until_idle()

    def test_ignores_other_model_or_size(self) -> None:
        self.index(self.save_doc())
        with db.connect() as conn, mock.patch("app.rag.query.embed", side_effect=fake_embed):
            self.assertTrue(query._vector_search(conn, "roads"))
            with mock.patch.dict("os.environ", {"OLLAMA_EMBED_DIM": "1024"}):
                self.assertEqual(query._vector_search(conn, "roads"), [])

    def test_embedding_failure_raises(self) -> None:
        self.index(self.save_doc())
        with db.connect() as conn, mock.patch("app.rag.query.embed", side_effect=httpx.ConnectError("down")):
            with self.assertRaises(httpx.ConnectError):
                query._vector_search(conn, "roads")


def ollama_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(body), request=httpx.Request("POST", "http://ollama"))


@mock.patch("app.llm.time.sleep")
class OllamaEmbedTest(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict("os.environ", {"OLLAMA_EMBED_MODEL": "qwen3-embedding:latest", "OLLAMA_EMBED_DIM": "2"})
        env.start()
        self.addCleanup(env.stop)

    def test_prefix_and_dimensions(self, _sleep: mock.Mock) -> None:
        with mock.patch("app.llm.httpx.post", return_value=ollama_response(200, {"embeddings": [[0.1, 0.2]]})) as post:
            llm.embed(["What caused it?"], "RETRIEVAL_QUERY")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["dimensions"], 2)
        self.assertTrue(body["input"][0].startswith("Instruct: "))
        self.assertTrue(body["input"][0].endswith("Query: What caused it?"))

    def test_documents_have_no_prefix(self, _sleep: mock.Mock) -> None:
        with mock.patch("app.llm.httpx.post", return_value=ollama_response(200, {"embeddings": [[0.1, 0.2]]})) as post:
            llm.embed(["The road closed."], "RETRIEVAL_DOCUMENT")
        self.assertEqual(post.call_args.kwargs["json"]["input"], ["The road closed."])

    def test_retries_then_succeeds(self, sleep: mock.Mock) -> None:
        down = httpx.ConnectError("connection refused")
        ok = ollama_response(200, {"embeddings": [[0.1, 0.2]]})
        with mock.patch("app.llm.httpx.post", side_effect=[down, down, down, ok]) as post:
            with self.assertLogs("app.llm", "WARNING"):
                llm.embed(["x"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(post.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8])

    def test_gives_up_after_retries(self, _sleep: mock.Mock) -> None:
        with mock.patch("app.llm.httpx.post", side_effect=httpx.ConnectError("refused")) as post:
            with self.assertLogs("app.llm", "WARNING"), self.assertRaises(httpx.ConnectError):
                llm.embed(["x"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(post.call_count, 4)

    def test_client_error_is_not_retried(self, _sleep: mock.Mock) -> None:
        missing = ollama_response(404, {"error": 'model "qwen3-embedding" not found, try pulling it first'})
        with mock.patch("app.llm.httpx.post", return_value=missing) as post:
            with self.assertRaisesRegex(llm.OllamaError, "not found"):
                llm.embed(["x"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(post.call_count, 1)

    def test_unknown_model_is_rejected(self, _sleep: mock.Mock) -> None:
        with mock.patch.dict("os.environ", {"OLLAMA_EMBED_MODEL": "some-other-embedder"}):
            with self.assertRaisesRegex(RuntimeError, "EMBED_PREFIXES"):
                llm.embed(["x"], "RETRIEVAL_DOCUMENT")

    def test_model_key(self, _sleep: mock.Mock) -> None:
        self.assertEqual(llm.embed_model_key(), "qwen3-embedding:latest@2")
        with mock.patch.dict("os.environ", {"OLLAMA_EMBED_DIM": ""}):
            self.assertEqual(llm.embed_model_key(), "qwen3-embedding:latest@full")


if __name__ == "__main__":
    unittest.main()
