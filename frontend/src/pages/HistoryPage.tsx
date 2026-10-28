import { useEffect, useState } from 'react'
import { getDocument, listDocuments, type DocumentDetail, type DocumentSummary } from '../api/client'
import { describeIndex, INDEX_POLL_MS, isIndexing } from '../indexStatus'

interface Props {
  onOpen: (doc: DocumentDetail) => void
}

export default function HistoryPage({ onOpen }: Props) {
  const [docs, setDocs] = useState<DocumentSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listDocuments().then(setDocs).catch((e: Error) => setError(e.message))
  }, [])

  // Refresh while any document is still indexing, so its Index column moves on its own.
  const anyIndexing = docs.some((d) => isIndexing(d.index_status))
  useEffect(() => {
    if (!anyIndexing) return
    const timer = setInterval(() => {
      listDocuments().then(setDocs).catch(() => {})
    }, INDEX_POLL_MS)
    return () => clearInterval(timer)
  }, [anyIndexing])

  async function open(id: string) {
    try {
      onOpen(await getDocument(id))
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div>
      <h2>History</h2>
      {error && <p role="alert">{error}</p>}
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th>Parser</th>
            <th>Status</th>
            <th>Index</th>
            <th>Uploaded</th>
            <th>Saved</th>
          </tr>
        </thead>
        <tbody>
          {docs.map((d) => (
            <tr key={d.id} onClick={() => open(d.id)} style={{ cursor: 'pointer' }}>
              <td>{d.original_filename}</td>
              <td>{d.parser_name} {d.parser_version}</td>
              <td>{d.status}</td>
              <td>{d.status === 'saved' ? describeIndex(d.index_status) : '-'}</td>
              <td>{d.uploaded_at}</td>
              <td>{d.saved_at ?? '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
