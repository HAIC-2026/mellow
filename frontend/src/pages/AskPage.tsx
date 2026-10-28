import { useState, type FormEvent } from 'react'
import { ask, type AskResponse, type Citation, type GraphDebug } from '../api/client'
import SourceViewer from './SourceViewer'
import GraphCanvas, { type CanvasEdgeData } from '../components/GraphCanvas'
import './AskPage.css'

function formatJson(raw: string | null): string {
  if (raw === null) return '(LLM not called: nothing retrieved)'
  try {
    return JSON.stringify(JSON.parse(raw), null, 2)
  } catch {
    return raw
  }
}

// Splits a segment's span ids into runs of neighbouring spans. The indexer inserts spans
// in document order, so consecutive ids under the same doc, page and heading are adjacent
// in the source.
function groupCitations(ids: number[], citations: Record<string, Citation>): Citation[][] {
  const groups: Citation[][] = []
  for (const id of [...ids].sort((a, b) => a - b)) {
    const c = citations[id]
    const prev = groups.at(-1)?.at(-1)
    if (
      prev &&
      c.span_id === prev.span_id + 1 &&
      c.doc_id === prev.doc_id &&
      c.page_number === prev.page_number &&
      c.heading_path === prev.heading_path
    ) {
      groups.at(-1)!.push(c)
    } else {
      groups.push([c])
    }
  }
  return groups
}

function groupKey(group: Citation[]): string {
  return group.map((c) => c.span_id).join(',')
}

// The kept chains and conflicts from debug.graph: what the answer was built from.
function pathGraph(graph: GraphDebug | null | undefined) {
  if (!graph || graph.paths.length === 0) return null
  const byId = new Map(graph.assertions.map((a) => [a.id, a]))
  const nodes = new Map<number, { id: number; name: string; type: string; size: number }>()
  const edges = new Map<string, CanvasEdgeData>()
  const addLink = (src: number, dst: number, srcName: string, dstName: string, type: string, negative: boolean) => {
    nodes.set(src, { id: src, name: srcName, type: 'entity', size: 22 })
    nodes.set(dst, { id: dst, name: dstName, type: 'entity', size: 22 })
    const key = `${src}-${dst}-${type}-${negative ? 'neg' : 'pos'}`
    edges.set(key, {
      key, src, dst, label: type, causal: true, negative, conflict: false, review: false, width: 2,
    })
  }
  for (const path of graph.paths) {
    for (const link of path.links) {
      const first = byId.get(link[0])
      if (first) addLink(first.src, first.dst, first.src_name, first.dst_name, first.type, false)
    }
  }
  for (const c of graph.conflicts) {
    const first = byId.get(c.assertion_ids[0])
    if (first) addLink(c.src, c.dst, first.src_name, first.dst_name, c.type, true)
  }
  return { nodes: [...nodes.values()], edges: [...edges.values()], centre: graph.target?.id }
}

function quote(group: Citation[]): string {
  return group.map((c) => c.text).join(' ')
}

export default function AskPage() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<AskResponse | null>(null)
  const [selected, setSelected] = useState<Citation[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [debug, setDebug] = useState(false)
  const [showGraph, setShowGraph] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError(null)
    setSelected(null)
    try {
      setResult(await ask(question, debug))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // One chip per group of consecutive spans; number groups by first appearance so chips read [1], [2], ...
  const segmentGroups = result?.answer_segments.map((seg) => groupCitations(seg.citations, result.citations)) ?? []
  const groups = new Map<string, Citation[]>()
  segmentGroups.flat().forEach((g) => groups.has(groupKey(g)) || groups.set(groupKey(g), g))
  const numbers = new Map([...groups.keys()].map((key, i) => [key, i + 1]))
  const selectedKey = selected && groupKey(selected)

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
        <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input type="checkbox" checked={debug} onChange={(e) => setDebug(e.target.checked)} />
          Debug
        </label>
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
                {segmentGroups[i].map((g) => {
                  const key = groupKey(g)
                  return (
                    <button
                      key={key}
                      type="button"
                      className={`cite-chip${selectedKey === key ? ' active' : ''}`}
                      title={`${g[0].original_filename} › ${g[0].heading_path || g[0].page_title}\n“${quote(g)}”`}
                      onClick={() => setSelected(g)}
                    >
                      {numbers.get(key)}
                    </button>
                  )
                })}
              </p>
            ))}
            {groups.size > 0 && (
              <ol className="cite-list">
                {[...groups].map(([key, g]) => (
                  <li key={key} onClick={() => setSelected(g)}>
                    <strong>{g[0].original_filename}</strong> › {g[0].heading_path || g[0].page_title}
                    <div className="cite-quote">“{quote(g)}”</div>
                  </li>
                ))}
              </ol>
            )}
            {result.debug && (
              <div className="ask-debug">
                {pathGraph(result.debug.graph) && (
                  <details open>
                    <summary>Graph ({result.debug.graph?.paths.length} chain(s), {result.debug.graph?.conflicts.length} conflict(s))</summary>
                    <button onClick={() => setShowGraph((v) => !v)}>{showGraph ? 'Hide graph' : 'Show graph'}</button>
                    {showGraph && (
                      <GraphCanvas
                        nodes={pathGraph(result.debug.graph)!.nodes}
                        edges={pathGraph(result.debug.graph)!.edges}
                        centreId={pathGraph(result.debug.graph)!.centre}
                      />
                    )}
                  </details>
                )}
                <details open>
                  <summary>Chunks sent to the LLM ({result.debug.chunks.length})</summary>
                  {result.debug.chunks.map((c) => (
                    <details key={c.chunk_id}>
                      <summary>
                        #{c.fused_rank} · chunk {c.chunk_id} · keyword {c.keyword_rank ?? '-'} · vector{' '}
                        {c.vector_rank ?? '-'} · {c.original_filename} › {c.heading_path || c.page_title}
                      </summary>
                      <pre>{c.text}</pre>
                    </details>
                  ))}
                </details>
                <details>
                  <summary>Passages in the prompt</summary>
                  <pre>{result.debug.passages ?? '(LLM not called: nothing retrieved)'}</pre>
                </details>
                <details open>
                  <summary>Raw LLM response</summary>
                  <pre>{formatJson(result.debug.raw_response)}</pre>
                </details>
              </div>
            )}
          </div>
          {selected && (
            <div className="ask-source">
              <SourceViewer key={selectedKey} citations={selected} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
