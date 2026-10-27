export type DocumentStatus = "parsed" | "saved" | "failed";

export interface DocumentSummary {
  id: string;
  original_filename: string;
  extension: string;
  parser_name: string;
  parser_version: string;
  status: DocumentStatus;
  sha256: string;
  uploaded_at: string;
  saved_at: string | null;
  source: string | null;
}

export interface Page {
  number: number;
  title: string;
  markdown: string;
}

export interface DocumentDetail extends DocumentSummary {
  pages: Page[];
}

export interface ParserInfo {
  extension: string;
  parser_name: string;
  parser_version: string;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export function getParsers(): Promise<ParserInfo[]> {
  return request("/api/parsers");
}

export function listDocuments(): Promise<DocumentSummary[]> {
  return request("/api/documents");
}

export function getDocument(id: string): Promise<DocumentDetail> {
  return request(`/api/documents/${id}`);
}

export function uploadDocument(file: File): Promise<DocumentDetail> {
  const form = new FormData();
  form.append("file", file);
  return request("/api/documents", { method: "POST", body: form });
}

export function createManualDocument(
  title: string,
  source: string,
): Promise<DocumentDetail> {
  return request("/api/documents/manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, source }),
  });
}

export function llmEditSheet(
  id: string,
  sheetMarkdown: string,
  instruction: string,
): Promise<{ markdown: string }> {
  return request(`/api/documents/${id}/llm-edit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sheet_markdown: sheetMarkdown, instruction }),
  });
}

export interface Citation {
  span_id: number;
  doc_id: string;
  original_filename: string;
  page_number: number;
  page_title: string;
  heading_path: string;
  kind: "heading" | "sentence" | "table_row";
  // Offsets into the page markdown: page.markdown.slice(start, end) === text
  start: number;
  end: number;
  text: string;
}

export interface AnswerSegment {
  text: string;
  citations: number[];
}

export interface DebugChunk {
  chunk_id: number;
  doc_id: string;
  original_filename: string;
  page_number: number;
  page_title: string;
  heading_path: string;
  text: string;
  fused_rank: number;
  keyword_rank: number | null;
  vector_rank: number | null;
}

export interface AskDebug {
  chunks: DebugChunk[];
  // Exact passages text in the prompt; null when nothing was retrieved and the LLM was not called.
  passages: string | null;
  raw_response: string | null;
}

export interface AskResponse {
  found: boolean;
  answer_segments: AnswerSegment[];
  citations: Record<string, Citation>;
  debug?: AskDebug;
}

export function ask(question: string, debug = false): Promise<AskResponse> {
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, debug }),
  });
}

export function saveDocument(
  id: string,
  pages: Page[],
): Promise<DocumentDetail> {
  return request(`/api/documents/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pages }),
  });
}
