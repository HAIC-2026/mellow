import { useEffect, useRef, useState } from "react";
import {
  createManualDocument,
  getParsers,
  uploadDocument,
  type DocumentDetail,
} from "../api/client";

interface Props {
  onUploaded: (doc: DocumentDetail) => void;
}

export default function UploadPage({ onUploaded }: Props) {
  const [accept, setAccept] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [source, setSource] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getParsers()
      .then((parsers) => setAccept(parsers.map((p) => p.extension).join(",")))
      .catch((e: Error) => setError(e.message));
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      onUploaded(await uploadDocument(file));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function handleManual() {
    setBusy(true);
    setError(null);
    try {
      onUploaded(await createManualDocument(title || "Untitled", source));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <form onSubmit={handleSubmit}>
        <h2>Upload file</h2>
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          hidden
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          {file ? `Selected: ${file.name}` : "Choose file"}
        </button>
        <button type="submit" disabled={!file || busy}>
          {busy ? "Parsing..." : "Upload and parse"}
        </button>
        {error && <p role="alert">{error}</p>}
      </form>
      <h2>Or write markdown</h2>
      <label>
        Title
        <input value={title} onChange={(e) => setTitle(e.target.value)} />
      </label>
      <label>
        Source
        <input
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="URL, document, or person"
        />
      </label>
      <button type="button" onClick={handleManual} disabled={busy}>
        New blank document
      </button>
    </>
  );
}
