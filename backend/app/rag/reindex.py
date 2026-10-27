"""Rebuild the retrieval index for every saved document: python -m app.rag.reindex"""

from app.db import connect, init_db
from app.rag.indexer import index_document
from app.routers.documents import _read_pages


def main() -> None:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, original_filename, final_md_path FROM documents WHERE status = 'saved'"
        ).fetchall()
    for row in rows:
        warning = index_document(row["id"], _read_pages(row["final_md_path"]))
        print(f"{row['original_filename']}: {warning or 'ok'}")


if __name__ == "__main__":
    main()
