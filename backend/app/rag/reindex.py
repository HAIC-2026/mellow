"""Rebuild the retrieval index: python -m app.rag.reindex [--failed]

Queues a job per saved document, then works through the queue in this process.
A running server's worker may take some of the jobs; each job runs once either way.
"""

import argparse

from app.db import connect, init_db
from app.rag import jobs
from app.rag.worker import run_until_idle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--failed", action="store_true", help="only documents whose last job failed")
    args = parser.parse_args()

    init_db()
    query = "SELECT d.id FROM documents d"
    if args.failed:
        query += " JOIN index_jobs j ON j.doc_id = d.id WHERE j.status = 'failed' AND d.status = 'saved'"
    else:
        query += " WHERE d.status = 'saved'"
    with connect() as conn:
        doc_ids = [row["id"] for row in conn.execute(query)]
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
