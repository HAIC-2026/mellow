import logging
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_EMBED_MODEL = "qwen3-embedding"
DEFAULT_OLLAMA_EXTRACT_MODEL = "qwen3:30b"
EMBED_BATCH_SIZE = 64
OLLAMA_TIMEOUT_SECONDS = 600
# Waits between attempts: 4 attempts in total, then the error is raised.
OLLAMA_RETRY_WAITS = (2, 4, 8)

# Local embedding models take a text prefix instead of a task type. A wrong or missing
# prefix lowers retrieval quality without any error, so unknown models are rejected.
EMBED_PREFIXES = {
    "qwen3-embedding": {
        "RETRIEVAL_DOCUMENT": "",
        "RETRIEVAL_QUERY": "Instruct: Given a question, retrieve passages that answer it\nQuery: ",
    },
}

ANSWER_PROMPT = """You answer questions using only the numbered source passages below.

Rules:
- Use only facts stated in the passages. Do not use outside knowledge.
- Split the answer into segments. Each segment is one or two sentences.
- Every segment must list the ids of the passages that support it in span_ids.
  Cite the specific passages, not every passage.
- If the passages do not contain the answer, set found to false and return one
  segment saying the documents do not cover it, with empty span_ids.
{rules}
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


def embed_model_key() -> str:
    """Model and size of the vectors embed() returns, e.g. qwen3-embedding@1024.

    Stored with every vector. Vectors are only compared when their keys match: two models,
    or one model at two sizes, give incomparable vectors.
    """
    dim = os.environ.get("OLLAMA_EMBED_DIM")
    return f"{_embed_model()}@{dim or 'full'}"


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    # task_type: RETRIEVAL_DOCUMENT for indexed chunks, RETRIEVAL_QUERY for questions.
    model = _embed_model()
    prefixes = EMBED_PREFIXES.get(model.split(":")[0])
    if prefixes is None:
        raise RuntimeError(f"no embedding prefixes known for {model}; add them to EMBED_PREFIXES")
    prefix = prefixes[task_type]
    body: dict[str, Any] = {"model": model}
    # Unset means the model's full size. Smaller sizes are not unit length;
    # vector search divides by the norms, so that is fine.
    dim = os.environ.get("OLLAMA_EMBED_DIM")
    if dim:
        body["dimensions"] = int(dim)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = [prefix + t for t in texts[start : start + EMBED_BATCH_SIZE]]
        response = _ollama_post("/api/embed", {**body, "input": batch})
        vectors.extend(response.get("embeddings") or [])
    if len(vectors) != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def _embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", DEFAULT_OLLAMA_EMBED_MODEL)


def extract_model() -> str:
    return os.environ.get("OLLAMA_EXTRACT_MODEL", DEFAULT_OLLAMA_EXTRACT_MODEL)


def generate_json(prompt: str, model_cls: type[T], think: bool | None = None) -> T:
    """Ask the extraction model for JSON shaped like model_cls, and validate it.

    The schema is enforced by Ollama. A reply that still fails validation counts as a failed
    attempt and is retried like a server error.
    """
    if think is None:
        think = os.environ.get("OLLAMA_EXTRACT_THINK", "true").lower() not in ("0", "false", "no", "")
    body: dict[str, Any] = {
        "model": extract_model(),
        "messages": [{"role": "user", "content": prompt}],
        "format": model_cls.model_json_schema(),
        "stream": False,
        "think": think,
        "options": {"temperature": 0},
    }

    def attempt() -> T:
        content = _ollama_once("/api/chat", body)["message"]["content"]
        try:
            return model_cls.model_validate_json(content)
        except ValidationError as exc:
            raise InvalidOutput(f"{model_cls.__name__}: {exc}") from exc

    return _with_retry(attempt, retry_on=(InvalidOutput,))


def _ollama_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    return _with_retry(lambda: _ollama_once(path, body))


def _ollama_once(path: str, body: dict[str, Any]) -> dict[str, Any]:
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/") + path
    return _raise_for_status(httpx.post(url, json=body, timeout=OLLAMA_TIMEOUT_SECONDS))


def _raise_for_status(response: httpx.Response) -> dict[str, Any]:
    if response.is_error:
        # Ollama puts the reason in {"error": "..."}, e.g. a model that is not pulled.
        try:
            detail = response.json().get("error") or response.text
        except ValueError:
            detail = response.text
        raise OllamaError(response.status_code, detail)
    return response.json()


class OllamaError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"Ollama HTTP {status_code}: {detail}")
        self.status_code = status_code


class InvalidOutput(RuntimeError):
    """The model replied, but the reply does not match the schema."""


def _with_retry(call: Callable[[], T], retry_on: tuple[type[Exception], ...] = ()) -> T:
    # Retry what can recover on its own: Ollama not running yet, timeouts, server errors,
    # and replies that fail validation. Client errors such as a model that is not pulled fail at once.
    for wait in (*OLLAMA_RETRY_WAITS, None):
        try:
            return call()
        except (httpx.TransportError, OllamaError, *retry_on) as exc:
            if wait is None or (isinstance(exc, OllamaError) and exc.status_code < 500):
                raise
            log.warning("Ollama request failed (%s); retrying in %ss", exc, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def answer_with_citations(question: str, passages: str, rules: str = "") -> tuple[Answer, str]:
    """Returns the parsed answer and the model's raw response text. rules: extra instructions, e.g. graph chains."""
    client = genai.Client()
    response = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        contents=ANSWER_PROMPT.format(question=question, passages=passages, rules=f"\n{rules}\n" if rules else ""),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Answer,
            temperature=0,
        ),
    )
    raw = response.text or ""
    if isinstance(response.parsed, Answer):
        return response.parsed, raw
    return Answer.model_validate_json(raw), raw


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
