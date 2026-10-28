"""Rebuild the retrieval index: python -m app.rag.reindex [--failed | --graph]

Queues a job per saved document, then works through the queue in this process.
A running server's worker may take some of the jobs; each job runs once either way.
--graph queues only the extract stage, for documents whose chunks are current: it back-fills
documents indexed before the graph existed, or re-extracts after the relation types change.
"""

import argparse

from app.db import connect, init_db
from app.rag import jobs
from app.rag.worker import run_until_idle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--failed", action="store_true", help="only documents whose last job failed")
    parser.add_argument("--graph", action="store_true", help="only the extract stage, for current documents")
    args = parser.parse_args()
    if args.failed and args.graph:
        parser.error("--failed and --graph cannot be combined")

    init_db()
    query = "SELECT d.id FROM documents d"
    if args.failed:
        query += " JOIN index_jobs j ON j.doc_id = d.id WHERE j.status = 'failed' AND d.status = 'saved'"
    elif args.graph:
        # Stage embed done means the chunks are current. A job still embedding or failed
        # at embed has old chunks, so it is skipped.
        query += (
            " JOIN index_jobs j ON j.doc_id = d.id WHERE d.status = 'saved' AND d.indexed_at IS NOT NULL"
            " AND (j.stage = 'extract' OR (j.stage = 'embed' AND j.status = 'done'))"
        )
    else:
        query += " WHERE d.status = 'saved'"
    with connect() as conn:
        doc_ids = [row["id"] for row in conn.execute(query)]
    if args.graph:
        jobs.enqueue_graph(doc_ids)
    else:
        jobs.enqueue(doc_ids)
    run_until_idle()

    with connect() as conn:
        rows = conn.execute(
            "SELECT d.original_filename, j.status, j.error FROM index_jobs j JOIN documents d ON d.id = j.doc_id "
            f"WHERE j.doc_id IN ({','.join('?' * len(doc_ids))})",
            doc_ids,
        ).fetchall() if doc_ids else []
    for row in rows:
        print(f"{row['original_filename']}: {row['error'] or row['status']}")


if __name__ == "__main__":
    main()
