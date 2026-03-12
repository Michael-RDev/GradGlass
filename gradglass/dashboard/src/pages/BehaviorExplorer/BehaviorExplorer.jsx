import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { fetchPredictions } from '../../api'
import { LoadingSpinner, ErrorMessage } from '../../components/ui'

const FILTERS = [
  { id: 'all',       label: 'All',         cls: 'bg-slate-800 text-slate-300' },
  { id: 'errors',    label: 'Errors',      cls: 'bg-red-500/10 text-red-400 border border-red-500/20' },
  { id: 'flips',     label: 'Label Flips', cls: 'bg-amber-500/10 text-amber-400 border border-amber-500/20' },
  { id: 'improved',  label: 'Improved',    cls: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' },
]

export default function BehaviorExplorer() {
  const { runId } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedExample, setSelectedExample] = useState(null)
  const [filterMode, setFilterMode] = useState('all')

  useEffect(() => {
    fetchPredictions(runId)
      .then(res => {
        setData(res.predictions || [])
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!data || data.length === 0) return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <h1 className="text-3xl font-bold text-white mb-2">Behavior Explorer</h1>
      <div className="card text-center py-16">
        <p className="text-slate-400">No prediction data found.</p>
        <p className="text-sm text-slate-500 mt-2">Log predictions with <code className="font-mono bg-slate-800 px-1 rounded">run.log_batch(x, y, y_pred)</code></p>
      </div>
    </div>
  )

  const latest = data[data.length - 1]
  const previous = data.length > 1 ? data[data.length - 2] : null

  // Build example objects
  const allExamples = []
  if (latest.y_true && latest.y_pred) {
    for (let i = 0; i < latest.y_true.length; i++) {
      const isCorrect = latest.y_true[i] === latest.y_pred[i]
      let prevCorrect = null
      if (previous?.y_true && previous?.y_pred && previous.y_true.length > i) {
        prevCorrect = previous.y_true[i] === previous.y_pred[i]
      }
      const isFlip = prevCorrect !== null && prevCorrect !== isCorrect
      const isImproved = prevCorrect === false && isCorrect
      allExamples.push({
        id: i,
        true_label: latest.y_true[i],
        pred_label: latest.y_pred[i],
        isCorrect,
        prevCorrect,
        isFlip,
        isImproved,
        confidence: latest.confidence?.[i] ?? null,
        prevConfidence: previous?.confidence?.[i] ?? null,
      })
    }
  }

  const examples = allExamples.filter(ex => {
    if (filterMode === 'errors')   return !ex.isCorrect
    if (filterMode === 'flips')    return ex.isFlip
    if (filterMode === 'improved') return ex.isImproved
    return true
  })

  function statusBadge(ex) {
    if (ex.isImproved)
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-emerald-400/10 text-emerald-400 border border-emerald-400/20">Improved</span>
    if (ex.prevCorrect === true && !ex.isCorrect)
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-400/10 text-red-400 border border-red-400/20">Regressed</span>
    if (!ex.isCorrect)
      return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-400/10 text-orange-400 border border-orange-400/20">Error</span>
    return <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-800/50 text-slate-400 border border-slate-700/50">Stable</span>
  }

  const sel = selectedExample

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-indigo-400 bg-clip-text text-transparent mb-2">
            Behavior Explorer
          </h1>
          <p className="text-slate-400 text-sm">Analyze individual sample predictions, errors, and confidence changes.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[75vh]">
        {/* Main Grid List */}
        <div className="col-span-12 lg:col-span-8 card !p-0 overflow-hidden flex flex-col">
          <div className="p-4 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
            <h3 className="font-semibold text-slate-200">
              Samples <span className="text-slate-500 font-normal text-sm ml-1">({examples.length})</span>
            </h3>
            <div className="flex gap-2 text-xs">
              {FILTERS.map(f => (
                <button
                  key={f.id}
                  onClick={() => setFilterMode(f.id)}
                  className={`px-3 py-1.5 rounded-lg transition ${f.cls} ${filterMode === f.id ? 'ring-2 ring-white/20' : 'hover:opacity-80'}`}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          <div className="overflow-y-auto flex-1 custom-scrollbar">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-900 sticky top-0 border-b border-slate-800 text-slate-500">
                <tr>
                  <th className="p-4 font-medium">Index</th>
                  <th className="p-4 font-medium">True Label</th>
                  <th className="p-4 font-medium">Prediction</th>
                  <th className="p-4 font-medium hidden md:table-cell">Confidence</th>
                  <th className="p-4 font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {examples.map(ex => (
                  <tr
                    key={ex.id}
                    onClick={() => setSelectedExample(ex.id === sel?.id ? null : ex)}
                    className={`transition-colors cursor-pointer group ${
                      sel?.id === ex.id
                        ? 'bg-indigo-500/10 border-l-2 border-indigo-500'
                        : 'hover:bg-slate-800/50'
                    }`}
                  >
                    <td className="p-4 font-mono text-slate-500">#{ex.id}</td>
                    <td className="p-4 text-slate-300">{ex.true_label}</td>
                    <td className={`p-4 font-medium ${ex.isCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                      {ex.pred_label}
                    </td>
                    <td className="p-4 hidden md:table-cell">
                      {ex.confidence !== null ? (
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-indigo-500" style={{ width: `${ex.confidence * 100}%` }} />
                          </div>
                          <span className="text-xs text-slate-400 font-mono">{(ex.confidence * 100).toFixed(0)}%</span>
                        </div>
                      ) : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="p-4">{statusBadge(ex)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Inspector Panel */}
        <div className="col-span-12 lg:col-span-4">
          {sel ? (
            <div className="card h-full flex flex-col gap-5 overflow-y-auto custom-scrollbar">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                <h3 className="font-semibold text-slate-200">Sample #{sel.id}</h3>
                <button onClick={() => setSelectedExample(null)} className="text-slate-500 hover:text-slate-300 text-xs">✕ close</button>
              </div>

              {/* Labels */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-slate-800/60 rounded-lg p-3 text-center">
                  <div className="text-xs text-slate-500 mb-1">True Label</div>
                  <div className="text-2xl font-bold text-slate-200">{sel.true_label}</div>
                </div>
                <div className={`rounded-lg p-3 text-center ${sel.isCorrect ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                  <div className="text-xs text-slate-500 mb-1">Predicted</div>
                  <div className={`text-2xl font-bold ${sel.isCorrect ? 'text-emerald-400' : 'text-red-400'}`}>{sel.pred_label}</div>
                </div>
              </div>

              {/* Verdict */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">Verdict</span>
                {statusBadge(sel)}
              </div>

              {/* Confidence */}
              {sel.confidence !== null && (
                <div>
                  <div className="text-xs text-slate-500 mb-2">Confidence</div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-indigo-500 transition-all" style={{ width: `${sel.confidence * 100}%` }} />
                    </div>
                    <span className="text-sm font-mono text-slate-300 w-10 text-right">{(sel.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}

              {/* Confidence shift vs previous probe */}
              {sel.prevConfidence !== null && sel.confidence !== null && (
                <div>
                  <div className="text-xs text-slate-500 mb-1">Confidence shift vs previous checkpoint</div>
                  {(() => {
                    const delta = sel.confidence - sel.prevConfidence
                    const pos = delta >= 0
                    return (
                      <span className={`text-sm font-mono font-semibold ${pos ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pos ? '+' : ''}{(delta * 100).toFixed(1)}%
                      </span>
                    )
                  })()}
                </div>
              )}

              {/* Previous checkpoint status */}
              {sel.prevCorrect !== null && (
                <div>
                  <div className="text-xs text-slate-500 mb-1">Previous checkpoint</div>
                  <span className={`text-sm ${sel.prevCorrect ? 'text-emerald-400' : 'text-red-400'}`}>
                    {sel.prevCorrect ? '✓ Correct' : '✗ Wrong'}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="card h-full flex flex-col items-center justify-center text-center px-8 border-dashed border-slate-700">
              <div className="w-16 h-16 rounded-full bg-slate-800 mb-4 flex items-center justify-center text-slate-500">
                <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-slate-300">Inspector</h3>
              <p className="text-sm text-slate-500 mt-2">Click a sample row to view its label, confidence, and shift from the previous checkpoint.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

