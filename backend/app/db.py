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

-- One indexing job per document. Each save increments generation; a run whose generation
-- is no longer current discards its results.
CREATE TABLE IF NOT EXISTS index_jobs (
  doc_id      TEXT PRIMARY KEY,
  generation  INTEGER NOT NULL,
  stage       TEXT NOT NULL CHECK (stage IN ('embed', 'extract')),
  status      TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
  attempts    INTEGER NOT NULL DEFAULT 0,
  error       TEXT,
  updated_at  TEXT NOT NULL
);

-- Graph index (docs/graph-rag-v2.md, sections 1 and 3). Rows are rebuilt per document
-- by the extract stage. Entities are not shared across documents until resolution (phase 3).
CREATE TABLE IF NOT EXISTS entities (
  id              INTEGER PRIMARY KEY,
  canonical_name  TEXT NOT NULL,
  type            TEXT NOT NULL,
  description     TEXT,
  embedding       BLOB,
  embed_model     TEXT
);

CREATE TABLE IF NOT EXISTS entity_aliases (
  entity_id  INTEGER NOT NULL,
  alias      TEXT NOT NULL,
  PRIMARY KEY (entity_id, alias)
);

CREATE TABLE IF NOT EXISTS entity_mentions (
  id            INTEGER PRIMARY KEY,
  entity_id     INTEGER NOT NULL,
  span_id       INTEGER NOT NULL,
  doc_id        TEXT NOT NULL,
  surface_form  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entity_mentions_doc_id ON entity_mentions (doc_id);
CREATE INDEX IF NOT EXISTS entity_mentions_entity_id ON entity_mentions (entity_id);

CREATE TABLE IF NOT EXISTS relation_assertions (
  id               INTEGER PRIMARY KEY,
  src_entity       INTEGER NOT NULL,
  dst_entity       INTEGER NOT NULL,
  type             TEXT NOT NULL,
  predicate        TEXT NOT NULL,
  doc_id           TEXT NOT NULL,
  chunk_id         INTEGER NOT NULL,
  polarity         TEXT NOT NULL CHECK (polarity IN ('positive', 'negative')),
  explicit         INTEGER NOT NULL CHECK (explicit IN (0, 1)),
  extractor_model  TEXT NOT NULL,
  types_version    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS relation_assertions_doc_id ON relation_assertions (doc_id);
CREATE INDEX IF NOT EXISTS relation_assertions_src ON relation_assertions (src_entity);
CREATE INDEX IF NOT EXISTS relation_assertions_dst ON relation_assertions (dst_entity);

CREATE TABLE IF NOT EXISTS assertion_spans (
  assertion_id  INTEGER NOT NULL,
  span_id       INTEGER NOT NULL,
  PRIMARY KEY (assertion_id, span_id)
);

-- The relation type vocabulary. class is fixed; 'proposed' rows wait for review.
CREATE TABLE IF NOT EXISTS relation_types (
  name         TEXT PRIMARY KEY,
  description  TEXT NOT NULL,
  class        TEXT NOT NULL CHECK (class IN ('causal', 'structural', 'associative', 'other')),
  status       TEXT NOT NULL CHECK (status IN ('active', 'proposed')),
  inverse_of   TEXT
);

-- Raw wording -> type, for types that are not active. swap = 1 reverses src and dst.
CREATE TABLE IF NOT EXISTS predicate_map (
  predicate  TEXT PRIMARY KEY,
  type       TEXT NOT NULL,
  swap       INTEGER NOT NULL DEFAULT 0 CHECK (swap IN (0, 1))
);

CREATE TABLE IF NOT EXISTS index_meta (
  key    TEXT PRIMARY KEY,
  value  TEXT NOT NULL
);
"""

# Initial vocabulary (docs/graph-rag-v2.md, 1.3). Seeded once: INSERT OR IGNORE keeps later edits.
RELATION_TYPE_SEED = [
    ("causes", "src directly causes dst", "causal"),
    ("contributes_to", "src is one cause among several", "causal"),
    ("prevents", "src stops or reduces dst; reported as context, not traversed", "associative"),
    ("correlated_with", "stated association", "associative"),
    ("part_of", "src is a component, program or unit of dst", "structural"),
    ("responsible_for", "person or agency src owns or runs dst", "structural"),
    ("funds", "src provides money to dst", "structural"),
    ("located_in", "src is in area dst", "structural"),
]


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
        if "graph_indexed_at" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN graph_indexed_at TEXT")
        chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
        if "embed_model" not in chunk_columns:
            # Existing rows keep NULL, so vector search ignores them until the next reindex.
            conn.execute("ALTER TABLE chunks ADD COLUMN embed_model TEXT")
        # Older databases have chunks_fts without stemming. Rebuild it from chunks; no re-embedding needed.
        fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunks_fts'").fetchone()["sql"]
        if "porter" not in fts_sql:
            conn.execute("DROP TABLE chunks_fts")
            conn.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, tokenize='porter unicode61')")
            conn.execute("INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks")
        conn.executemany(
            "INSERT OR IGNORE INTO relation_types (name, description, class, status, inverse_of) "
            "VALUES (?, ?, ?, 'active', NULL)",
            RELATION_TYPE_SEED,
        )
        conn.execute("INSERT OR IGNORE INTO index_meta (key, value) VALUES ('relation_types_version', '1')")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
