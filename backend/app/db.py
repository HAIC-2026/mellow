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
  saved_at           TEXT
);
"""


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
