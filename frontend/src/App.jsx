import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

function Metric({ label, value }) {
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return <div className="space-y-1">
    <div className="flex justify-between gap-3 text-xs text-gray-300"><span>{label.replaceAll('_', ' ')}</span><span>{Number(value || 0).toFixed(2)}</span></div>
    <div className="h-1.5 rounded-full bg-white/10"><div className="h-1.5 rounded-full bg-orange-400" style={{ width: `${pct}%` }} /></div>
  </div>;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [query, setQuery] = useState('');
  const [strategy, setStrategy] = useState('auto');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState('Ready');
  const [latencySensitive, setLatencySensitive] = useState(false);

  const onUpload = async () => {
    if (!file || uploading) return;
    const formData = new FormData();
    formData.append('file', file);
    setUploading(true);
    setStatus('Extracting, chunking and indexing…');
    try {
      const res = await axios.post(`${API_BASE}/upload`, formData);
      setStatus(`Indexed ${res.data.chunks_indexed} chunks from ${res.data.filename}`);
    } catch (error) {
      setStatus(error.response?.data?.detail || 'Upload failed. Check the backend logs.');
    } finally {
      setUploading(false);
    }
  };

  const onChat = async () => {
    const question = query.trim();
    if (!question || loading) return;
    setMessages(prev => [...prev, { role: 'user', text: question }]);
    setQuery('');
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/chat`, {
        query: question,
        strategy,
        latency_sensitive: latencySensitive,
      });
      setMessages(prev => [...prev, { role: 'ai', text: res.data.answer, routing: res.data.routing }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: error.response?.data?.detail || 'Error reaching backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black text-white">
      <aside className="flex w-80 shrink-0 flex-col border-r border-white/10 bg-white/[0.04] p-6 backdrop-blur-xl">
        <h1 className="mb-2 bg-gradient-to-r from-orange-300 to-red-400 bg-clip-text text-2xl font-bold text-transparent">Adaptive GraphRAG</h1>
        <p className="mb-8 text-xs leading-5 text-gray-400">AHP-weighted routing · Hybrid retrieval · Explainable AI</p>

        <section className="space-y-3">
          <label className="block text-sm font-medium text-gray-200">Upload a document</label>
          <input type="file" accept=".pdf,.docx,.txt,.md,.csv,.json,.py,.js,.jsx,.html,.css,.xml,.png,.jpg,.jpeg,.webp,.mp3,.wav,.mp4"
            onChange={e => setFile(e.target.files?.[0] || null)}
            className="block w-full text-xs text-gray-300 file:mr-3 file:rounded-lg file:border-0 file:bg-orange-600 file:px-3 file:py-2 file:text-white hover:file:bg-orange-500" />
          <button onClick={onUpload} disabled={!file || uploading} className="w-full rounded-lg bg-orange-600 px-4 py-2.5 text-sm font-semibold transition hover:bg-orange-500 disabled:cursor-not-allowed disabled:opacity-50">
            {uploading ? 'Indexing…' : 'Upload and index'}
          </button>
          <p className="break-words text-xs text-gray-400">{status}</p>
        </section>

        <div className="my-6 border-t border-white/10" />
        <section className="space-y-3">
          <label htmlFor="strategy" className="block text-sm font-medium text-gray-200">Retrieval strategy</label>
          <select id="strategy" value={strategy} onChange={e => setStrategy(e.target.value)} className="w-full rounded-lg border border-white/10 bg-gray-900 px-3 py-2.5 text-sm text-white outline-none focus:border-orange-400">
            <option value="auto">Automatic (AHP + weighted matrix)</option>
            <option value="way1">Way 1 — Graph, then vector fallback</option>
            <option value="way2">Way 2 — Graph + vector in parallel</option>
          </select>
          <label className="flex items-center gap-2 text-xs text-gray-300">
            <input type="checkbox" checked={latencySensitive} onChange={e => setLatencySensitive(e.target.checked)} className="accent-orange-500" />
            Prefer lower latency/cost
          </label>
        </section>

        <div className="mt-auto border-t border-white/10 pt-5 text-xs leading-5 text-gray-500">
          Way 1 retrieves from Neo4j first and uses Qdrant if needed. Way 2 runs both retrieval paths concurrently and merges evidence.
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="border-b border-white/10 px-8 py-5">
          <h2 className="text-lg font-semibold">Document intelligence workspace</h2>
          <p className="mt-1 text-sm text-gray-400">Ask questions and inspect why a retrieval strategy was selected.</p>
        </header>

        <div className="flex-1 space-y-6 overflow-y-auto px-5 py-8 md:px-8">
          {messages.length === 0 && <div className="mx-auto mt-24 max-w-lg text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-orange-400/20 bg-orange-400/10 text-2xl">✦</div>
            <h3 className="text-xl font-semibold">Start with your documents</h3>
            <p className="mt-2 text-sm leading-6 text-gray-400">Upload a file, choose a retrieval strategy, then ask a question. Automatic mode explains its decision using graph coverage, vector relevance, query intent and document breadth.</p>
          </div>}

          {messages.map((message, index) => <div key={`${index}-${message.role}`} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`w-full max-w-3xl rounded-2xl px-5 py-4 shadow-lg ${message.role === 'user' ? 'max-w-2xl bg-gradient-to-r from-orange-600 to-red-600' : 'border border-white/10 bg-white/[0.045]'}`}>
              <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-white/60">{message.role === 'user' ? 'You' : 'RAG assistant'}</div>
              <p className="whitespace-pre-wrap text-sm leading-6">{message.text}</p>
              {message.routing && <div className="mt-5 space-y-4 border-t border-white/10 pt-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h4 className="text-sm font-semibold">Explainable routing decision</h4>
                  <span className="rounded-full border border-orange-400/30 bg-orange-400/10 px-3 py-1 text-xs font-semibold text-orange-200">{message.routing.actual_strategy_path}</span>
                </div>
                <p className="text-xs leading-5 text-gray-300">{message.routing.explanation}</p>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-white/10 bg-black/20 p-3"><p className="mb-2 text-xs font-semibold text-gray-300">Weighted suitability</p><p className="text-sm">Way 1 <b>{message.routing.scores?.way1}</b></p><p className="text-sm">Way 2 <b>{message.routing.scores?.way2}</b></p></div>
                  <div className="rounded-xl border border-white/10 bg-black/20 p-3"><p className="mb-2 text-xs font-semibold text-gray-300">AHP consistency</p><p className="text-sm">CR: <b>{message.routing.ahp_consistency_ratio}</b></p><p className="text-xs text-gray-400">{message.routing.ahp_consistency_acceptable ? 'Within the commonly used 0.10 threshold.' : 'Weights need consistency review.'}</p></div>
                </div>
                <div className="space-y-3 rounded-xl border border-white/10 bg-black/20 p-3">
                  <p className="text-xs font-semibold text-gray-300">Observed routing metrics</p>
                  {Object.entries(message.routing.metrics || {}).map(([key, value]) => <Metric key={key} label={key} value={value} />)}
                </div>
                {message.routing.reasons?.length > 0 && <div><p className="mb-2 text-xs font-semibold text-gray-300">Why</p><ul className="list-disc space-y-1 pl-5 text-xs leading-5 text-gray-400">{message.routing.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul></div>}
                {message.routing.retrieved_chunks?.length > 0 && <details className="rounded-xl border border-white/10 p-3"><summary className="cursor-pointer text-xs font-semibold text-gray-300">Retrieved document evidence ({message.routing.retrieved_chunks.length})</summary><div className="mt-3 space-y-3">{message.routing.retrieved_chunks.map((chunk, i) => <div key={i} className="border-l-2 border-orange-400/50 pl-3"><p className="text-xs font-semibold">{chunk.source}{chunk.page ? ` · page ${chunk.page}` : ''}</p><p className="mt-1 text-xs leading-5 text-gray-400">{chunk.preview}</p></div>)}</div></details>}
              </div>}
            </div>
          </div>)}
          {loading && <div className="text-sm text-gray-400">Retrieving evidence and generating an answer…</div>}
        </div>

        <form onSubmit={e => { e.preventDefault(); onChat(); }} className="border-t border-white/10 bg-white/[0.03] p-4 md:p-6">
          <div className="mx-auto flex max-w-4xl items-center gap-3">
            <input value={query} onChange={e => setQuery(e.target.value)} placeholder="Ask about entities, relationships, or uploaded documents…" className="min-w-0 flex-1 rounded-xl border border-white/10 bg-white/[0.04] px-4 py-3 text-sm outline-none placeholder:text-gray-500 focus:border-orange-400" />
            <button type="submit" disabled={loading || !query.trim()} className="rounded-xl bg-orange-600 px-5 py-3 text-sm font-semibold transition hover:bg-orange-500 disabled:opacity-50">{loading ? 'Thinking…' : 'Send'}</button>
          </div>
        </form>
      </main>
    </div>
  );
}
