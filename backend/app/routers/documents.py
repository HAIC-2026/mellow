from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import DATA_DIR, MARKDOWN_DIR, UPLOADS_DIR
from app.db import connect
from app.llm import edit_sheet
from app.parsers.registry import PARSERS, get_parser
from app.storage import new_doc_id, sha256_of

router = APIRouter(prefix="/api")

COLUMNS = (
    "id, original_filename, extension, parser_name, parser_version, "
    "status, sha256, uploaded_at, saved_at"
)


class SaveRequest(BaseModel):
    markdown: str


class LlmEditRequest(BaseModel):
    sheet_markdown: str
    instruction: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_markdown(rel_path: str | None) -> str | None:
    if rel_path is None:
        return None
    return (DATA_DIR / rel_path).read_text(encoding="utf-8")


def _fetch_row(doc_id: str) -> Any:
    with connect() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "document not found")
    return row


def _payload(row: Any, markdown: str | None) -> dict[str, Any]:
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
        "markdown": markdown,
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
    markdown: str | None = None
    status = "parsed"
    try:
        markdown = parser.parse(raw_path)
        generated_rel = f"markdown/{doc_id}/generated.md"
        (DATA_DIR / generated_rel).parent.mkdir(parents=True)
        (DATA_DIR / generated_rel).write_text(markdown, encoding="utf-8")
    except Exception as exc:
        status = "failed"
        _insert(doc_id, original_name, extension, parser, raw_rel, None, data, status)
        raise HTTPException(422, f"parse failed: {exc}") from exc

    _insert(doc_id, original_name, extension, parser, raw_rel, generated_rel, data, status)
    return _payload(_fetch_row(doc_id), markdown)


def _insert(
    doc_id: str,
    original_name: str,
    extension: str,
    parser: Any,
    raw_rel: str,
    generated_rel: str | None,
    data: bytes,
    status: str,
) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, original_filename, extension, parser_name, "
            "parser_version, raw_path, generated_md_path, final_md_path, sha256, status, "
            "uploaded_at, saved_at) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL)",
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
    return _payload(row, _read_markdown(rel))


@router.put("/documents/{doc_id}")
def save_document(doc_id: str, body: SaveRequest) -> dict[str, Any]:
    row = _fetch_row(doc_id)
    if row["status"] == "failed":
        raise HTTPException(409, "cannot save a document that failed to parse")

    final_rel = f"markdown/{doc_id}/final.md"
    final_path = DATA_DIR / final_rel
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(body.markdown, encoding="utf-8")

    saved_at = _now()
    with connect() as conn:
        conn.execute(
            "UPDATE documents SET final_md_path = ?, status = 'saved', saved_at = ? WHERE id = ?",
            (final_rel, saved_at, doc_id),
        )
    return _payload(_fetch_row(doc_id), body.markdown)


@router.post("/documents/{doc_id}/llm-edit")
def llm_edit(doc_id: str, body: LlmEditRequest) -> dict[str, str]:
    _fetch_row(doc_id)
    try:
        revised = edit_sheet(body.sheet_markdown, body.instruction)
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc
    return {"markdown": revised}
