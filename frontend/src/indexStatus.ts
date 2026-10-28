import type { IndexStatus } from "./api/client";

export const INDEX_POLL_MS = 2000;

export function isIndexing(status: IndexStatus | null): boolean {
  return status?.status === "pending" || status?.status === "running";
}

export function describeIndex(status: IndexStatus | null): string {
  if (status === null) return "Not indexed";
  if (status.status === "failed") return `Failed: ${status.error ?? "unknown error"}`;
  if (status.status === "done") return "Ready";
  return status.stage === "embed" ? "Indexing..." : "Extracting...";
}
