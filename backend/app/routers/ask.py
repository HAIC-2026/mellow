from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.query import ask

router = APIRouter(prefix="/api")


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=20)
    # Adds "debug": the chunks sent to the LLM, the exact passages text and the raw LLM response.
    debug: bool = False


@router.post("/ask")
def ask_question(body: AskRequest) -> dict[str, Any]:
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "question is empty")
    try:
        return ask(question, body.top_k, body.debug)
    except Exception as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc
