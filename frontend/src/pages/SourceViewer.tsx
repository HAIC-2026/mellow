import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getDocument, type Citation, type DocumentDetail } from '../api/client'

interface Props {
  // Adjacent spans from one page and heading, in source order.
  citations: Citation[]
}

interface HastNode {
  type: string
  tagName?: string
  value?: string
  properties?: Record<string, unknown>
  children?: HastNode[]
  position?: { start: { offset?: number }; end: { offset?: number } }
}

// Wraps the cited [start, end) source range in <mark>, using the source offsets
// that remark/rehype keep on every node. Table rows get a row highlight instead.
function rehypeCite(range: { start: number; end: number }) {
  return (tree: HastNode) => {
    const walk = (node: HastNode) => {
      if (!node.children) return
      node.children = node.children.flatMap((child) => {
        const s = child.position?.start.offset
        const e = child.position?.end.offset
        const overlaps = s !== undefined && e !== undefined && s < range.end && e > range.start
        if (child.type === 'element' && child.tagName === 'tr' && overlaps) {
          child.properties = { ...child.properties, className: ['cite-row'] }
        }
        if (child.type !== 'text' || !overlaps || !child.value) {
          walk(child)
          return [child]
        }
        const value = child.value
        // Escapes or entities make value and source lengths differ; then mark the whole node.
        const exact = value.length === e! - s!
        const a = exact ? Math.max(0, range.start - s!) : 0
        const b = exact ? Math.min(value.length, range.end - s!) : value.length
        const mark: HastNode = {
          type: 'element',
          tagName: 'mark',
          properties: { className: ['cite'] },
          children: [{ type: 'text', value: value.slice(a, b) }],
        }
        return [
          ...(a > 0 ? [{ type: 'text', value: value.slice(0, a) }] : []),
          mark,
          ...(b < value.length ? [{ type: 'text', value: value.slice(b) }] : []),
        ]
      })
    }
    walk(tree)
  }
}

export default function SourceViewer({ citations }: Props) {
  const [doc, setDoc] = useState<DocumentDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const citation = citations[0]
  const range = { start: citation.start, end: citations[citations.length - 1].end }

  useEffect(() => {
    getDocument(citation.doc_id).then(setDoc).catch((e: Error) => setError(e.message))
  }, [citation.doc_id])

  const page = doc?.pages.find((p) => p.number === citation.page_number)
  // Check each span on its own: the text between spans is not stored.
  const stale = page !== undefined && citations.some((c) => page.markdown.slice(c.start, c.end) !== c.text)

  useEffect(() => {
    ref.current?.querySelector('mark.cite, tr.cite-row')?.scrollIntoView({ block: 'center' })
  }, [page, citations])

  return (
    <div>
      <h3 style={{ marginBottom: 4 }}>{citation.original_filename}</h3>
      <div style={{ color: '#666', marginBottom: 8 }}>
        {citation.page_title}
        {citation.heading_path && ` › ${citation.heading_path}`}
      </div>
      {error && <p role="alert">{error}</p>}
      {stale && <p role="alert">This document changed since the answer was generated; the highlight may be off.</p>}
      {page && (
        <div ref={ref} className="md-preview source-viewer">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[[rehypeCite, range]]}>
            {page.markdown}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}
