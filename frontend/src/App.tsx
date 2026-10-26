import { useState } from 'react'
import type { DocumentDetail } from './api/client'
import HistoryPage from './pages/HistoryPage'
import ReviewPage from './pages/ReviewPage'
import UploadPage from './pages/UploadPage'

type View = 'upload' | 'review' | 'history'

export default function App() {
  const [view, setView] = useState<View>('upload')
  const [doc, setDoc] = useState<DocumentDetail | null>(null)

  function openDoc(next: DocumentDetail) {
    setDoc(next)
    setView('review')
  }

  return (
    <main style={{ maxWidth: 1100, margin: '0 auto', padding: 16 }}>
      <nav style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <button onClick={() => setView('upload')}>Upload</button>
        <button onClick={() => setView('history')}>History</button>
      </nav>
      {view === 'upload' && <UploadPage onUploaded={openDoc} />}
      {view === 'review' && doc && <ReviewPage key={doc.id} doc={doc} onSaved={setDoc} />}
      {view === 'history' && <HistoryPage onOpen={openDoc} />}
    </main>
  )
}
