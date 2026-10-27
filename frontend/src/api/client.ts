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
