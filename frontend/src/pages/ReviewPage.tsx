import { useState } from 'react'
import { diffLines, type Change } from 'diff'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './ReviewPage.css'
import { llmEditSheet, saveDocument, type DocumentDetail } from '../api/client'

interface Props {
  doc: DocumentDetail
  onSaved: (doc: DocumentDetail) => void
}

function splitSheets(markdown: string): string[] {
  return markdown.split(/^(?=## )/m)
}

function sheetLabel(chunk: string, index: number): string {
  if (!chunk.startsWith('## ')) return 'Intro'
  return chunk.split('\n')[0].slice(3).trim() || `Sheet ${index + 1}`
}

export default function ReviewPage({ doc, onSaved }: Props) {
  const [baseline, setBaseline] = useState(doc.markdown ?? '')
  const [sections, setSections] = useState(() => splitSheets(baseline))
  const [active, setActive] = useState(0)
  const [changes, setChanges] = useState<Change[] | null>(null)
  const [busy, setBusy] = useState(false)
  const [instruction, setInstruction] = useState('')
  const [llmBusy, setLlmBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const current = sections.join('')
  const hasChanges = changes?.some((c) => c.added || c.removed) ?? false

  function updateActive(value: string) {
    setSections((prev) => prev.map((s, i) => (i === active ? value : s)))
    setChanges(null)
  }

  async function applyLlmEdit() {
    setLlmBusy(true)
    setError(null)
    setMessage(null)
    try {
      const res = await llmEditSheet(doc.id, sections[active] ?? '', instruction)
      updateActive(res.markdown)
      setMessage('LLM edit applied to this sheet. Preview changes before saving.')
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLlmBusy(false)
    }
  }

  function previewChanges() {
    setChanges(diffLines(baseline, current))
    setMessage(null)
    setError(null)
  }

  async function handleSave() {
    setBusy(true)
    setError(null)
    setMessage(null)
    try {
      const saved = await saveDocument(doc.id, current)
      setBaseline(saved.markdown ?? '')
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
        Parser: {doc.parser_name} {doc.parser_version} | Status: {doc.status}
      </p>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        {sections.map((chunk, i) => (
          <button
            key={i}
            onClick={() => setActive(i)}
            style={{ fontWeight: i === active ? 'bold' : 'normal' }}
          >
            {sheetLabel(chunk, i)}
          </button>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <textarea
          value={sections[active] ?? ''}
          onChange={(e) => updateActive(e.target.value)}
          rows={30}
          style={{ width: '100%', fontFamily: 'monospace' }}
        />
        <div className="md-preview" style={{ height: '100%', maxHeight: 720, overflow: 'auto', border: '1px solid #ccc', padding: 8 }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{sections[active] ?? ''}</ReactMarkdown>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <input
          type="text"
          value={instruction}
          onChange={(e) => setInstruction(e.target.value)}
          placeholder="Describe a change for this sheet, e.g. fix typos in column headers"
          style={{ flex: 1 }}
        />
        <button onClick={applyLlmEdit} disabled={llmBusy || busy || !instruction.trim()}>
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
