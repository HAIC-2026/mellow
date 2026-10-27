import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from app.config import DATA_DIR, DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  id                 TEXT PRIMARY KEY,
  original_filename  TEXT NOT NULL,
  extension          TEXT NOT NULL,
  parser_name        TEXT NOT NULL,
  parser_version     TEXT NOT NULL,
  raw_path           TEXT NOT NULL,
  generated_md_path  TEXT,
  final_md_path      TEXT,
  sha256             TEXT NOT NULL,
  status             TEXT NOT NULL CHECK (status IN ('parsed', 'saved', 'failed')),
  uploaded_at        TEXT NOT NULL,
  saved_at           TEXT,
  source             TEXT
);

-- Retrieval index over saved documents. Rebuilt per document on every save.
CREATE TABLE IF NOT EXISTS chunks (
  id            INTEGER PRIMARY KEY,
  doc_id        TEXT NOT NULL,
  page_number   INTEGER NOT NULL,
  page_title    TEXT NOT NULL,
  heading_path  TEXT NOT NULL,
  text          TEXT NOT NULL,
  embedding     BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);

-- Citable units. page markdown[start:end] == text.
CREATE TABLE IF NOT EXISTS spans (
  id            INTEGER PRIMARY KEY,
  chunk_id      INTEGER NOT NULL,
  doc_id        TEXT NOT NULL,
  page_number   INTEGER NOT NULL,
  kind          TEXT NOT NULL,
  start         INTEGER NOT NULL,
  end           INTEGER NOT NULL,
  text          TEXT NOT NULL,
  llm_text      TEXT NOT NULL,
  heading_path  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS spans_doc_id ON spans (doc_id);
CREATE INDEX IF NOT EXISTS spans_chunk_id ON spans (chunk_id);

-- rowid = chunks.id. Porter stemming so "agency" matches "agencies".
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS skips existing tables, so add newer columns here.
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        if "source" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN source TEXT")
        if "indexed_at" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN indexed_at TEXT")
        # Older databases have chunks_fts without stemming. Rebuild it from chunks; no re-embedding needed.
        fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()["sql"]
        if "porter" not in fts_sql:
            conn.execute("DROP TABLE chunks_fts")
            conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='porter unicode61')")
            conn.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
