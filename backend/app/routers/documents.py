import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import DATA_DIR, MARKDOWN_DIR, UPLOADS_DIR
from app.db import connect
from app.llm import edit_sheet
from app.parsers.base import Page
from app.parsers.registry import MANUAL_PARSER, PARSERS, get_parser
from app.rag import jobs
from app.storage import new_doc_id, sha256_of

router = APIRouter(prefix="/api")

COLUMNS = (
    "id, original_filename, extension, parser_name, parser_version, "
    "status, sha256, uploaded_at, saved_at, source"
)


class PageModel(BaseModel):
    number: int
    title: str
    markdown: str


class SaveRequest(BaseModel):
    pages: list[PageModel]


class ManualRequest(BaseModel):
    title: str = "Untitled"
    source: str | None = None


class LlmEditRequest(BaseModel):
    sheet_markdown: str
    instruction: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _legacy_pages(markdown: str) -> list[dict[str, Any]]:
    # Documents parsed before paging stored one markdown file; split it per sheet.
    chunks = [c for c in re.split(r"^(?=## )", markdown, flags=re.MULTILINE) if c.strip()]
    pages = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk.split("\n")[0][3:].strip() if chunk.startswith("## ") else "Intro"
        pages.append({"number": i, "title": title or f"Sheet {i}", "markdown": chunk})
    return pages


def _read_pages(rel_path: str | None) -> list[dict[str, Any]]:
    if rel_path is None:
        return []
    path = DATA_DIR / rel_path
    if path.suffix == ".md":
        return _legacy_pages(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _write_pages(rel_path: str, pages: list[dict[str, Any]]) -> None:
    path = DATA_DIR / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")


def _fetch_row(doc_id: str) -> Any:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "document not found")
    return row


def _payload(row: Any, pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "original_filename": row["original_filename"],
        "extension": row["extension"],
        "parser_name": row["parser_name"],
        "parser_version": row["parser_version"],
        "status": row["status"],
        "sha256": row["sha256"],
        "uploaded_at": row["uploaded_at"],
        "saved_at": row["saved_at"],
        "source": row["source"],
        "pages": pages,
        # null for documents never queued: unsaved, or saved before indexing became a job.
        "index_status": jobs.status_of(row["id"]),
    }


@router.get("/parsers")
def list_parsers() -> list[dict[str, str]]:
    return [
        {"extension": ext, "parser_name": parser.name, "parser_version": parser.version}
        for ext, parser in PARSERS.items()
    ]


@router.post("/documents", status_code=201)
def upload_document(file: UploadFile) -> dict[str, Any]:
    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()
    parser = get_parser(extension)
    if parser is None:
        raise HTTPException(415, f"unsupported file type: {extension or 'none'}")

    data = file.file.read()
    doc_id = new_doc_id()
    raw_rel = f"uploads/{doc_id}/original{extension}"
    raw_path = DATA_DIR / raw_rel
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(data)

    generated_rel: str | None = None
    pages: list[dict[str, Any]] = []
    status = "parsed"
    try:
        parsed: list[Page] = parser.parse(raw_path)
        pages = [asdict(p) for p in parsed]
        generated_rel = f"markdown/{doc_id}/generated.json"
        _write_pages(generated_rel, pages)
    except Exception as exc:
        status = "failed"
        _insert(doc_id, original_name, extension, parser, raw_rel, None, data, status)
        raise HTTPException(422, f"parse failed: {exc}") from exc

    _insert(doc_id, original_name, extension, parser, raw_rel, generated_rel, data, status)
    return _payload(_fetch_row(doc_id), pages)


@router.post("/documents/manual", status_code=201)
def create_manual_document(body: ManualRequest) -> dict[str, Any]:
    # No source file: the page is typed or pasted in the review UI and stored via PUT.
    title = body.title.strip() or "Untitled"
    source = (body.source or "").strip() or None
    doc_id = new_doc_id()
    pages = [asdict(p) for p in MANUAL_PARSER.parse(Path(""))]
    pages[0]["title"] = title
    generated_rel = f"markdown/{doc_id}/generated.json"
    _write_pages(generated_rel, pages)
    _insert(doc_id, title, "", MANUAL_PARSER, "", generated_rel, b"", "parsed", source=source)
    return _payload(_fetch_row(doc_id), pages)


def _insert(
    doc_id: str,
    original_name: str,
    extension: str,
    parser: Any,
    raw_rel: str,
    generated_rel: str | None,
    data: bytes,
    status: str,
    source: str | None = None,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, original_filename, extension, parser_name, "
            "parser_version, raw_path, generated_md_path, final_md_path, sha256, status, "
            "uploaded_at, saved_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?)",
            (
                doc_id,
                original_name,
                extension,
                parser.name,
                parser.version,
                raw_rel,
                generated_rel,
                sha256_of(data),
                status,
                _now(),
                source,
            ),
        )


@router.get("/documents")
def list_documents() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(f"SELECT {COLUMNS} FROM documents ORDER BY uploaded_at DESC").fetchall()
    return [dict(row) for row in rows]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    row = _fetch_row(doc_id)
    rel = row["final_md_path"] if row["status"] == "saved" else row["generated_md_path"]
    return _payload(row, _read_pages(rel))


@router.put("/documents/{doc_id}")
def save_document(doc_id: str, body: SaveRequest) -> dict[str, Any]:
    row = _fetch_row(doc_id)
    if row["status"] == "failed":
        raise HTTPException(409, "cannot save a document that failed to parse")

    pages = [page.model_dump() for page in body.pages]
    final_rel = f"markdown/{doc_id}/final.json"
    _write_pages(final_rel, pages)

    saved_at = _now()
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET final_md_path = ?, status = 'saved', saved_at = ? WHERE id = ?",
            (final_rel, saved_at, doc_id),
        )
    # The save is durable at this point. Indexing runs in the background worker; the
    # response carries the job status, which GET /api/documents/{id} keeps updating.
    jobs.enqueue([doc_id])
    return _payload(_fetch_row(doc_id), pages)


@router.post("/documents/{doc_id}/llm-edit")
def llm_edit(doc_id: str, body: LlmEditRequest) -> dict[str, str]:
    _fetch_row(doc_id)
    try:
        revised = edit_sheet(body.sheet_markdown, body.instruction)
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc
    return {"markdown": revised}
