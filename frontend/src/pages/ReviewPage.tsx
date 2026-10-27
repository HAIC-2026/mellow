import { useState } from 'react'
import { diffLines, type Change } from 'diff'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ReviewPage.css'
import { llmEditSheet, saveDocument, type DocumentDetail, type Page } from '../api/client'

interface Props {
  doc: DocumentDetail
  onSaved: (doc: DocumentDetail) => void
}

function joinPages(pages: Page[]): string {
  return pages.map((p) => p.markdown).join('')
}

export default function ReviewPage({ doc, onSaved }: Props) {
  const [baseline, setBaseline] = useState<Page[]>(doc.pages)
  const [pages, setPages] = useState<Page[]>(doc.pages)
  const [active, setActive] = useState(0)
  const [changes, setChanges] = useState<Change[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [llmBusy, setLlmBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const activePage = pages[active]
  const hasChanges = changes?.some((c) => c.added || c.removed) ?? false

  function isDirty(i: number): boolean {
    return pages[i]?.markdown !== baseline[i]?.markdown
  }

  function updateActive(markdown: string) {
    setPages((prev) => prev.map((p, i) => (i === active ? { ...p, markdown } : p)))
    setChanges(null)
  }

  async function applyLlmEdit() {
    if (!activePage) return
    setLlmBusy(true)
    setError(null)
    setMessage(null)
    try {
      const res = await llmEditSheet(doc.id, activePage.markdown, instruction)
      updateActive(res.markdown)
      setMessage('LLM edit applied to this page. Preview changes before saving.')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLlmBusy(false)
    }
  }

  function previewChanges() {
    setChanges(diffLines(joinPages(baseline), joinPages(pages)))
    setMessage(null)
    setError(null)
  }

  async function handleSave() {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const saved = await saveDocument(doc.id, pages)
      setBaseline(saved.pages)
      setPages(saved.pages)
      setChanges(null)
      setMessage(`Saved at ${saved.saved_at}`)
      onSaved(saved)
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <h2>Review: {doc.original_filename}</h2>
      <p>
        Parser: {doc.parser_name} {doc.parser_version} | Status: {doc.status} | Pages: {pages.length}
      </p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        {pages.map((page, i) => (
          <button
            key={page.number}
            onClick={() => setActive(i)}
            style={{ fontWeight: i === active ? 'bold' : 'normal' }}
          >
            {page.title}
            {isDirty(i) ? ' *' : ''}
          </button>
        ))}
      </div>
      {activePage && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <textarea
            value={activePage.markdown}
            onChange={(e) => updateActive(e.target.value)}
            rows={30}
            style={{ width: '100%', fontFamily: 'monospace' }}
          />
          <div className="md-preview" style={{ height: '100%', maxHeight: 720, overflow: 'auto', border: '1px solid #ccc', padding: 8 }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{activePage.markdown}</ReactMarkdown>
          </div>
        </div>
      )}
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <input
          type="text"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          placeholder="Describe a change for this page, e.g. fix typos in column headers"
          style={{ flex: 1 }}
        />
        <button onClick={applyLlmEdit} disabled={llmBusy || busy || !instruction.trim() || !activePage}>
          {llmBusy ? 'Asking LLM...' : 'Apply with LLM'}
        </button>
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button onClick={previewChanges} disabled={busy}>
          Preview changes
        </button>
        <button onClick={handleSave} disabled={busy || changes === null || !hasChanges}>
          {busy ? 'Saving...' : 'Save'}
        </button>
      </div>
      {changes !== null && (
        <div style={{ marginTop: 12 }}>
          <h3>Changes</h3>
          {hasChanges ? (
            <pre style={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: 400, overflow: 'auto' }}>
              {changes.map((part, i) => (
                <span
                  key={i}
                  style={{
                    display: 'block',
                    background: part.added ? '#e6ffec' : part.removed ? '#ffebe9' : 'transparent',
                    color: part.removed ? '#82071e' : undefined,
                    textDecoration: part.removed ? 'line-through' : undefined,
                  }}
                >
                  {part.added ? '+ ' : part.removed ? '- ' : '  '}
                  {part.value}
                </span>
              ))}
            </pre>
          ) : (
            <p>No changes since last load or save.</p>
          )}
        </div>
      )}
      {message && <p>{message}</p>}
      {error && <p role="alert">{error}</p>}
    </div>
  )
}
