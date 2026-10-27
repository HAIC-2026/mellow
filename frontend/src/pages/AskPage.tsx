import { useState, type FormEvent } from 'react'
import { ask, type AskResponse, type Citation } from '../api/client'
import SourceViewer from './SourceViewer'
import './AskPage.css'

export default function AskPage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AskResponse | null>(null)
  const [selected, setSelected] = useState<Citation | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setSelected(null)
    try {
      setResult(await ask(question))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // Number citations by first appearance so chips read [1], [2], ...
  const numbers = new Map<number, number>()
  result?.answer_segments.forEach((seg) =>
    seg.citations.forEach((id) => numbers.has(id) || numbers.set(id, numbers.size + 1)),
  )

  return (
    <div>
      <h2>Ask</h2>
      <form onSubmit={submit} style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your saved documents"
          style={{ flex: 1 }}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Asking…' : 'Ask'}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {result && (
        <div className="ask-layout">
          <div className="ask-answer">
            {result.answer_segments.map((seg, i) => (
              <p key={i}>
                {seg.text}
                {seg.citations.map((id) => {
                  const c = result.citations[id]
                  return (
                    <button
                      key={id}
                      type="button"
                      className={`cite-chip${selected?.span_id === id ? ' active' : ''}`}
                      title={`${c.original_filename} › ${c.heading_path || c.page_title}\n“${c.text}”`}
                      onClick={() => setSelected(c)}
                    >
                      {numbers.get(id)}
                    </button>
                  )
                })}
              </p>
            ))}
            {numbers.size > 0 && (
              <ol className="cite-list">
                {[...numbers.keys()].map((id) => {
                  const c = result.citations[id]
                  return (
                    <li key={id} onClick={() => setSelected(c)}>
                      <strong>{c.original_filename}</strong> › {c.heading_path || c.page_title}
                      <div className="cite-quote">“{c.text}”</div>
                    </li>
                  )
                })}
              </ol>
            )}
          </div>
          {selected && (
            <div className="ask-source">
              <SourceViewer key={selected.span_id} citation={selected} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
