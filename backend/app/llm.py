import os
import time
from collections.abc import Callable
from typing import TypeVar

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

T = TypeVar("T")

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
EMBED_BATCH_SIZE = 100
EMBED_BATCH_CHARS = 30_000
RATE_LIMIT_RETRIES = 5
RATE_LIMIT_WAIT_SECONDS = 60

ANSWER_PROMPT = """You answer questions using only the numbered source passages below.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Split the answer into segments. Each segment is one or two sentences.
- Every segment must list the ids of the passages that support it in span_ids.
  Cite the specific passages, not every passage.
- If the passages do not contain the answer, set found to false and return one
  segment saying the documents do not cover it, with empty span_ids.

Question:
{question}

Passages:
{passages}
"""


class AnswerSegment(BaseModel):
    text: str
    span_ids: list[int]


class Answer(BaseModel):
    found: bool
    answer_segments: list[AnswerSegment]

PROMPT_TEMPLATE = """You edit one sheet of a Markdown document converted from an Excel workbook.

Apply the user's instruction to the sheet below. Keep the "## " sheet heading and all
unchanged content exactly as it is. Output only the complete revised Markdown for this
sheet, with no code fences and no commentary.

Instruction:
{instruction}

Sheet:
{sheet}
"""


def edit_sheet(sheet_markdown: str, instruction: str) -> str:
    client = genai.Client()
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        contents=PROMPT_TEMPLATE.format(instruction=instruction, sheet=sheet_markdown),
    )
    text = response.text or ""
    return _strip_code_fence(text)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    # task_type: RETRIEVAL_DOCUMENT for indexed chunks, RETRIEVAL_QUERY for questions.
    client = genai.Client()
    model = os.environ.get("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    vectors: list[list[float]] = []
    for batch in _batches(texts):
        response = _with_rate_limit_retry(
            lambda: client.models.embed_content(
                model=model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        )
        vectors.extend(e.values or [] for e in response.embeddings or [])
    if len(vectors) != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def _batches(texts: list[str]) -> list[list[str]]:
    # Bounded by count and size so one request stays well under the per-minute token quota.
    batches: list[list[str]] = [[]]
    size = 0
    for text in texts:
        if batches[-1] and (len(batches[-1]) >= EMBED_BATCH_SIZE or size + len(text) > EMBED_BATCH_CHARS):
            batches.append([])
            size = 0
        batches[-1].append(text)
        size += len(text)
    return batches


def _with_rate_limit_retry(call: Callable[[], T]) -> T:
    # Free-tier embedding quota is per minute; large workbooks exceed it in one save.
    for attempt in range(RATE_LIMIT_RETRIES):
        try:
            return call()
        except errors.ClientError as exc:
            if exc.code != 429 or attempt == RATE_LIMIT_RETRIES - 1:
                raise
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
    raise AssertionError("unreachable")


def answer_with_citations(question: str, passages: str) -> Answer:
    client = genai.Client()
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        contents=ANSWER_PROMPT.format(question=question, passages=passages),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Answer,
            temperature=0,
        ),
    )
    if isinstance(response.parsed, Answer):
        return response.parsed
    return Answer.model_validate_json(response.text or "")


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        return "\n".join(lines) + "\n"
    return text
