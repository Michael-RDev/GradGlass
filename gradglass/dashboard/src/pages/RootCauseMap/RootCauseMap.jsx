import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { fetchArchitecture, fetchGradients } from '../../api'
import { LoadingSpinner, ErrorMessage } from '../../components/ui'
import { AlertCircle, Target, Layers, TrendingDown, TrendingUp, Zap, Minus } from 'lucide-react'

const FLAG_META = {
  VANISHING:          { label: 'Vanishing Gradient',   color: 'blue',   icon: TrendingDown },
  EXPLODING:          { label: 'Exploding Gradient',   color: 'red',    icon: TrendingUp   },
  NOISY:              { label: 'Noisy Gradient',        color: 'amber',  icon: Zap          },
  DEAD:               { label: 'Dead Gradient',         color: 'slate',  icon: Minus        },
  DISTRIBUTION_SHIFT: { label: 'Distribution Shift',   color: 'violet', icon: AlertCircle  },
}

function flagBadge(flag) {
  const m = FLAG_META[flag] || { label: flag, color: 'slate', icon: AlertCircle }
  const Icon = m.icon
  const cls = {
    blue:   'bg-blue-500/10   text-blue-400   border-blue-500/20',
    red:    'bg-red-500/10    text-red-400    border-red-500/20',
    amber:  'bg-amber-500/10  text-amber-400  border-amber-500/20',
    slate:  'bg-slate-700/50  text-slate-400  border-slate-600/30',
    violet: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
  }[m.color]
  return (
    <span key={flag} className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border ${cls}`}>
      <Icon className="w-3 h-3" /> {m.label}
    </span>
  )
}

export default function RootCauseMap() {
  const { runId } = useParams()
  const [data, setData] = useState({ gradients: null, architecture: null })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedIssue, setSelectedIssue] = useState(null)

  useEffect(() => {
    Promise.all([
      fetchArchitecture(runId).catch(() => null),
      fetchGradients(runId).catch(() => null),
    ])
      .then(([arch, grads]) => {
        setData({ architecture: arch, gradients: grads })
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />

  const flaggedLayers = (data.gradients?.analysis || []).filter(l => l.flags.length > 0)

  // All layers that share any flag with selectedIssue
  const suspectLayers = selectedIssue
    ? (data.gradients?.analysis || []).filter(l =>
        l.flags.some(f => selectedIssue.flags.includes(f))
      )
    : []

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-2">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
          Root Cause Map
        </h1>
      </div>

      <p className="text-slate-400">
        Trace the root cause of metric anomalies back to specific data slices and layer behavior.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 min-h-[60vh]">

        {/* Column 1: Symptoms */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Target className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-semibold text-slate-200">1. Symptoms</h2>
          </div>

          <div className="space-y-2">
            {flaggedLayers.length > 0 ? (
              flaggedLayers.map((issue, i) => {
                const isSelected = selectedIssue?.layer === issue.layer
                return (
                  <button
                    key={i}
                    onClick={() => setSelectedIssue(isSelected ? null : issue)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-indigo-500/15 border-indigo-500/40 ring-1 ring-indigo-500/30'
                        : 'bg-slate-800/40 border-slate-700/40 hover:bg-slate-800/70 hover:border-slate-600/60'
                    }`}
                  >
                    <div className="flex flex-wrap gap-1 mb-1.5">
                      {issue.flags.map(f => flagBadge(f))}
                    </div>
                    <div className="text-xs text-slate-400 font-mono truncate">{issue.layer}</div>
                    <div className="text-xs text-slate-500 mt-1">
                      norm {issue.grad_norm?.toExponential(2) ?? '—'} · mean {issue.grad_mean?.toExponential(2) ?? '—'}
                    </div>
                  </button>
                )
              })
            ) : (
              <div className="text-center py-10 opacity-50">
                <Target className="w-8 h-8 mx-auto text-slate-500 mb-2" />
                <p className="text-sm text-slate-400">No major gradient symptoms detected.</p>
              </div>
            )}
          </div>
        </div>

        {/* Column 2: Data Slices */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Layers className="w-5 h-5 text-emerald-400" />
            <h2 className="text-lg font-semibold text-slate-200">2. Affected Slices</h2>
          </div>

          <div className="flex flex-col items-center justify-center py-10 text-center gap-3">
            <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
              <Layers className="w-5 h-5 text-slate-500" />
            </div>
            <p className="text-sm text-slate-400 font-medium">Slice logging not configured</p>
            <p className="text-xs text-slate-500 max-w-[220px]">
              Use <code className="font-mono bg-slate-800 px-1 rounded">run.log_slice(name, y_true, y_pred)</code> during training to enable per-group analysis here.
            </p>
          </div>
        </div>

        {/* Column 3: Layer Suspects */}
        <div className="card space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
            <Network className="w-5 h-5 text-violet-400" />
            <h2 className="text-lg font-semibold text-slate-200">3. Layer Suspects</h2>
          </div>

          {selectedIssue ? (
            <div className="space-y-3">
              <p className="text-xs text-slate-500">Layers sharing flags with <span className="font-mono text-slate-300">{selectedIssue.layer}</span>:</p>
              {suspectLayers.map((layer, i) => (
                <div key={i} className="p-3 bg-slate-800/50 border border-slate-700/50 rounded-lg space-y-2">
                  <div className="text-xs font-mono text-slate-200 truncate">{layer.layer}</div>
                  <div className="flex flex-wrap gap-1">
                    {layer.flags.map(f => flagBadge(f))}
                  </div>
                  <div className="grid grid-cols-2 gap-x-4 text-xs text-slate-500">
                    <span>norm <span className="text-slate-300 font-mono">{layer.grad_norm?.toExponential(2) ?? '—'}</span></span>
                    <span>mean <span className="text-slate-300 font-mono">{layer.grad_mean?.toExponential(2) ?? '—'}</span></span>
                    <span>var  <span className="text-slate-300 font-mono">{layer.grad_var?.toExponential(2) ?? '—'}</span></span>
                    <span>kl   <span className="text-slate-300 font-mono">{layer.kl_div?.toFixed(3) ?? '—'}</span></span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-10 opacity-50">
              <Network className="w-8 h-8 mx-auto text-slate-500 mb-2" />
              <p className="text-sm text-slate-400">Select a symptom to identify responsible layers.</p>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}

function Network(props) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="16" y="16" width="6" height="6" rx="1"></rect>
      <rect x="2" y="16" width="6" height="6" rx="1"></rect>
      <rect x="9" y="2" width="6" height="6" rx="1"></rect>
      <path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"></path>
      <path d="M12 12V8"></path>
    </svg>
  )
}

