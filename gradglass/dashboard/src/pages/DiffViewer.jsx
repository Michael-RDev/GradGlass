import { useState, useEffect, useMemo } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import { fetchDiff, fetchCheckpoints } from '../api'
import { SeverityBadge, LoadingSpinner, ErrorMessage, MetricValue } from '../components/ui'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from 'recharts'
import { AlertTriangle, Layers, Search, Info, TrendingUp, TrendingDown, Minus, CheckCircle, XCircle, Zap, Eye } from 'lucide-react'

const SEVERITY_COLORS = {
  LOW: '#10b981',
  MEDIUM: '#f59e0b',
  HIGH: '#f97316',
  CRITICAL: '#ef4444',
}

const SEVERITY_ORDER = { LOW: 0, MEDIUM: 1, HIGH: 2, CRITICAL: 3 }

// Generate a plain-English narrative of what changed between checkpoints
function buildNarrative(diff, stepA, stepB) {
  if (!diff || !diff.summary) return null
  const { total_layers, critical = 0, high = 0, medium = 0, low = 0 } = diff.summary

  const lines = []

  // Overall health
  const problemCount = critical + high
  if (critical === 0 && high === 0) {
    lines.push({ icon: '✅', color: 'text-emerald-400', text: `Training looks stable — no critical or high-severity layer changes between step ${stepA} and step ${stepB}.` })
  } else if (critical > 0) {
    lines.push({ icon: '🚨', color: 'text-red-400', text: `${critical} layer${critical > 1 ? 's' : ''} changed dramatically (CRITICAL). This can mean catastrophic forgetting, an LR spike, or a data shift.` })
  } else {
    lines.push({ icon: '⚠️', color: 'text-orange-400', text: `${high} layer${high > 1 ? 's' : ''} changed significantly (HIGH). Review whether this matches your expectations.` })
  }

  // Layer change spread
  const unchangedCount = diff.layers?.filter(l => l.frob_norm < 0.001).length || 0
  if (unchangedCount > 0) {
    lines.push({ icon: '❄️', color: 'text-blue-400', text: `${unchangedCount} layer${unchangedCount > 1 ? 's' : ''} barely changed — these may be frozen, or receiving very weak gradients.` })
  }

  // Top movers
  const top = diff.layers ? [...diff.layers].sort((a, b) => b.frob_norm - a.frob_norm)[0] : null
  if (top) {
    lines.push({ icon: '📊', color: 'text-slate-300', text: `Biggest mover: "${top.name}" (Δ norm = ${top.frob_norm.toFixed(4)}, cosine similarity = ${top.cos_sim.toFixed(4)}).` })
  }

  // Cosine similarity tip
  const lowCos = diff.layers?.filter(l => l.cos_sim < 0.9) || []
  if (lowCos.length > 0) {
    lines.push({ icon: '🔄', color: 'text-amber-400', text: `${lowCos.length} layer${lowCos.length > 1 ? 's have' : ' has'} low cosine similarity (< 0.9) — the weight vectors rotated substantially, not just scaled.` })
  }

  return lines
}

// Tooltip explaining technical terms
function MetricHelp({ term, explanation }) {
  const [show, setShow] = useState(false)
  return (
    <span className="relative inline-block">
      <button onMouseEnter={() => setShow(true)} onMouseLeave={() => setShow(false)}
        className="text-slate-500 hover:text-slate-300 transition-colors ml-1 align-middle">
        <Info className="w-3 h-3 inline" />
      </button>
      {show && (
        <div className="absolute z-50 bottom-6 left-0 bg-slate-900 border border-slate-700 rounded-lg p-3 w-64 text-xs text-slate-300 shadow-xl pointer-events-none">
          <span className="font-semibold text-white">{term}: </span>{explanation}
        </div>
      )}
    </span>
  )
}

// Mini progress bar for severity levels
function SeverityBar({ counts, total }) {
  if (!total) return null
  const segments = [
    { key: 'critical', label: 'Critical', color: '#ef4444', count: counts.critical || 0 },
    { key: 'high', label: 'High', color: '#f97316', count: counts.high || 0 },
    { key: 'medium', label: 'Medium', color: '#f59e0b', count: counts.medium || 0 },
    { key: 'low', label: 'Low', color: '#10b981', count: counts.low || 0 },
  ]
  return (
    <div>
      <div className="flex h-2 rounded-full overflow-hidden gap-0.5 mb-2">
        {segments.map(s => s.count > 0 && (
          <div key={s.key} style={{ width: `${(s.count / total) * 100}%`, backgroundColor: s.color }} title={`${s.label}: ${s.count}`} />
        ))}
      </div>
      <div className="flex gap-3 text-xs">
        {segments.map(s => s.count > 0 && (
          <span key={s.key} style={{ color: s.color }} className="font-mono">{s.count} {s.label}</span>
        ))}
      </div>
    </div>
  )
}

export default function DiffViewer() {
  const { runId } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const [diff, setDiff] = useState(null)
  const [checkpoints, setCheckpoints] = useState([])
  const [stepA, setStepA] = useState(searchParams.get('a') || '')
  const [stepB, setStepB] = useState(searchParams.get('b') || '')
  const [selectedLayer, setSelectedLayer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [layerSearch, setLayerSearch] = useState('')
  const [severityFilter, setSeverityFilter] = useState('ALL')
  const [sortBy, setSortBy] = useState('frob_norm') // 'frob_norm' | 'cos_sim' | 'name' | 'severity'

  // Load checkpoints for selector
  useEffect(() => {
    fetchCheckpoints(runId)
      .then(data => {
        const ckpts = data.checkpoints || []
        setCheckpoints(ckpts)
        if (!stepA && !stepB && ckpts.length >= 2) {
          setStepA(String(ckpts[ckpts.length - 2].step))
          setStepB(String(ckpts[ckpts.length - 1].step))
        }
      })
      .catch(() => {})
  }, [runId])

  // Load diff when steps change
  useEffect(() => {
    if (stepA && stepB && stepA !== stepB) {
      setLoading(true)
      setError(null)
      fetchDiff(runId, parseInt(stepA), parseInt(stepB), true)
        .then(data => {
          setDiff(data)
          setLoading(false)
          if (data.layers?.length > 0) setSelectedLayer(data.layers[0])
        })
        .catch(err => {
          setError(err.message)
          setLoading(false)
        })
    }
  }, [runId, stepA, stepB])

  const narrative = useMemo(() => diff ? buildNarrative(diff, stepA, stepB) : null, [diff, stepA, stepB])

  const filteredLayers = useMemo(() => {
    if (!diff?.layers) return []
    let layers = diff.layers
    if (severityFilter !== 'ALL') layers = layers.filter(l => l.severity === severityFilter)
    if (layerSearch) layers = layers.filter(l => l.name.toLowerCase().includes(layerSearch.toLowerCase()))
    return [...layers].sort((a, b) => {
      if (sortBy === 'frob_norm') return b.frob_norm - a.frob_norm
      if (sortBy === 'cos_sim') return a.cos_sim - b.cos_sim
      if (sortBy === 'severity') return (SEVERITY_ORDER[b.severity] || 0) - (SEVERITY_ORDER[a.severity] || 0)
      return a.name.localeCompare(b.name)
    })
  }, [diff, severityFilter, layerSearch, sortBy])

  const severityCounts = useMemo(() => ({
    critical: diff?.summary?.critical || 0,
    high: diff?.summary?.high || 0,
    medium: diff?.summary?.medium || 0,
    low: diff?.summary?.low || 0,
  }), [diff])

  // Radar data for selected layer
  const radarData = selectedLayer ? [
    { metric: 'Change\nMagnitude', value: Math.min(selectedLayer.frob_norm / 2, 1) * 100 },
    { metric: 'Direction\nShift', value: (1 - selectedLayer.cos_sim) * 100 },
    { metric: '% Params\nChanged', value: (selectedLayer.percent_changed || 0) * 100 },
    { metric: 'Max\nDelta', value: Math.min(Math.abs(selectedLayer.max_delta) * 1000, 100) },
  ] : []

  return (
    <div>
      {/* Header with checkpoint selector */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Diff Explorer</h1>
          <p className="text-sm text-slate-500 mt-1">Understand what changed in your model's weights between two checkpoints</p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-500">Checkpoint A:</label>
          <select
            value={stepA}
            onChange={e => setStepA(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-glass-500"
          >
            <option value="">Select...</option>
            {checkpoints.map(c => (
              <option key={c.step} value={c.step}>
                Step {c.step?.toLocaleString()} {c.tag ? `(${c.tag})` : ''}
              </option>
            ))}
          </select>

          <span className="text-slate-600">→</span>

          <label className="text-xs text-slate-500">Checkpoint B:</label>
          <select
            value={stepB}
            onChange={e => setStepB(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-glass-500"
          >
            <option value="">Select...</option>
            {checkpoints.map(c => (
              <option key={c.step} value={c.step}>
                Step {c.step?.toLocaleString()} {c.tag ? `(${c.tag})` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}

      {diff && !loading && (
        <>
          {/* Plain-English Narrative */}
          {narrative && (
            <div className="card mb-4 bg-slate-900/60 border-slate-700/40">
              <div className="flex items-center gap-2 mb-3">
                <Eye className="w-4 h-4 text-glass-400" />
                <h3 className="text-sm font-semibold text-glass-300 uppercase tracking-wider">What Changed</h3>
                <span className="text-xs text-slate-600">Step {stepA} → Step {stepB}</span>
              </div>
              <div className="space-y-2">
                {narrative.map((line, i) => (
                  <div key={i} className="flex items-start gap-2">
                    <span className="text-base leading-tight">{line.icon}</span>
                    <p className={`text-sm ${line.color}`}>{line.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Severity breakdown bar */}
          <div className="card mb-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-slate-400">Layer Change Distribution</h3>
              <span className="text-xs text-slate-600">{diff.summary?.total_layers} layers total</span>
            </div>
            <SeverityBar counts={severityCounts} total={diff.summary?.total_layers || 0} />
            <p className="text-xs text-slate-600 mt-3">
              <span className="text-slate-400">How to read this:</span> Each segment shows how many layers changed by that severity.
              CRITICAL = dramatic rewrite. LOW = minor tuning. More low/medium = healthy gradient flow.
            </p>
          </div>

          <div className="grid grid-cols-12 gap-4">
            {/* Left panel: Layer tree */}
            <div className="col-span-4">
              {/* Layer filter / search */}
              <div className="card mb-3 bg-slate-900/40">
                <div className="flex gap-2 mb-2">
                  <div className="relative flex-1">
                    <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
                    <input type="text" placeholder="Search layers…" value={layerSearch}
                      onChange={e => setLayerSearch(e.target.value)}
                      className="w-full bg-slate-800 text-xs text-slate-300 rounded-lg pl-7 pr-2 py-1.5 border border-slate-700 focus:border-glass-500 focus:outline-none"
                    />
                  </div>
                  <select value={severityFilter} onChange={e => setSeverityFilter(e.target.value)}
                    className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-xs text-slate-300 focus:outline-none">
                    <option value="ALL">All</option>
                    <option value="CRITICAL">Critical</option>
                    <option value="HIGH">High</option>
                    <option value="MEDIUM">Medium</option>
                    <option value="LOW">Low</option>
                  </select>
                </div>
                <div className="flex gap-1">
                  {['frob_norm','severity','cos_sim','name'].map(s => (
                    <button key={s} onClick={() => setSortBy(s)}
                      className={`text-[10px] px-2 py-0.5 rounded font-mono transition-colors ${sortBy === s ? 'bg-glass-600/30 text-glass-300' : 'text-slate-500 hover:text-slate-300'}`}>
                      {s === 'frob_norm' ? '↓ΔNorm' : s === 'cos_sim' ? '↑Cos' : s === 'severity' ? 'Sev' : 'A-Z'}
                    </button>
                  ))}
                </div>
              </div>

              {/* Layer list */}
              <div className="card bg-slate-900/40">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Model Layers</h3>
                  <span className="text-xs text-slate-600">{filteredLayers.length} shown</span>
                </div>
                <div className="space-y-1.5 max-h-[560px] overflow-y-auto custom-scrollbar pr-1">
                  {filteredLayers.map((layer, i) => {
                    const isSelected = selectedLayer?.name === layer.name
                    const severityColor = SEVERITY_COLORS[layer.severity] || '#64748b'
                    return (
                      <button key={i} onClick={() => setSelectedLayer(layer)}
                        className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm border hover:-translate-y-0.5 ${
                          isSelected
                            ? 'bg-indigo-500/10 border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.1)]'
                            : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-800/80 hover:border-slate-600'
                        }`}
                      >
                        {/* Layer name + severity */}
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <span className={`font-mono text-xs truncate ${isSelected ? 'text-indigo-300' : 'text-slate-300'}`}>{layer.name}</span>
                          <SeverityBadge severity={layer.severity} />
                        </div>
                        {/* Mini visual: frob_norm bar */}
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
                            <div className="h-full rounded-full transition-all"
                              style={{ width: `${Math.min(layer.frob_norm / 2 * 100, 100)}%`, backgroundColor: severityColor }} />
                          </div>
                          <span className="text-[10px] font-mono text-slate-500 shrink-0">Δ{layer.frob_norm.toFixed(3)}</span>
                        </div>
                        {/* Cos sim indicator */}
                        <div className="flex items-center gap-1 mt-1">
                          <span className="text-[10px] text-slate-600">cos=</span>
                          <span className={`text-[10px] font-mono ${layer.cos_sim > 0.99 ? 'text-emerald-500' : layer.cos_sim > 0.9 ? 'text-amber-500' : 'text-red-400'}`}>
                            {layer.cos_sim.toFixed(4)}
                          </span>
                          {layer.cos_sim > 0.99 && <span className="text-[9px] text-emerald-600">similar direction</span>}
                          {layer.cos_sim < 0.9 && <span className="text-[9px] text-red-600">rotated!</span>}
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>
            </div>

            {/* Right panel: Layer details */}
            <div className="col-span-8">
              {selectedLayer ? (
                <>
                  {/* Layer header */}
                  <div className="card mb-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <h3 className="font-mono text-lg font-medium text-white">{selectedLayer.name}</h3>
                        <p className="text-xs text-slate-500 mt-1">
                          Shape: [{selectedLayer.shape?.join(' × ')}]
                          {selectedLayer.shape && <span className="ml-2 text-slate-600">({selectedLayer.shape.reduce((a,b) => a*b, 1).toLocaleString()} params)</span>}
                        </p>
                      </div>
                      <SeverityBadge severity={selectedLayer.severity} />
                    </div>

                    {/* Plain language explanation */}
                    <div className="mt-3 p-3 bg-slate-800/60 rounded-lg border border-slate-700/40">
                      <p className="text-xs text-slate-300">
                        {selectedLayer.severity === 'CRITICAL' && '🚨 This layer changed drastically. Weight vectors shifted in both magnitude and direction — unusual for a single training window.'}
                        {selectedLayer.severity === 'HIGH' && '⚠️ Significant update. The weights moved a lot, which is normal in early training but worth reviewing in later stages.'}
                        {selectedLayer.severity === 'MEDIUM' && '📈 Moderate update. The layer is learning actively and weights are moving in a healthy direction.'}
                        {selectedLayer.severity === 'LOW' && '✅ Minor update. This layer is either well-converged or receiving small gradients — typical for stable training.'}
                      </p>
                    </div>

                    <div className="grid grid-cols-4 gap-4 mt-4">
                      <div>
                        <p className="text-xs text-slate-500 mb-1">
                          Frobenius Norm Δ
                          <MetricHelp term="Frobenius Norm" explanation="The total 'distance' the weight matrix moved. Higher = bigger change. Like Euclidean distance for matrices." />
                        </p>
                        <p className="text-lg font-mono font-semibold text-white">{selectedLayer.frob_norm.toFixed(4)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 mb-1">
                          Cosine Similarity
                          <MetricHelp term="Cosine Similarity" explanation="How similar the weight direction is before vs after. 1.0 = same direction (just scaled). Near 0 = completely different direction." />
                        </p>
                        <p className={`text-lg font-mono font-semibold ${selectedLayer.cos_sim > 0.99 ? 'text-emerald-400' : selectedLayer.cos_sim > 0.9 ? 'text-amber-400' : 'text-red-400'}`}>
                          {selectedLayer.cos_sim.toFixed(4)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 mb-1">
                          % Params Changed
                          <MetricHelp term="% Changed" explanation="Fraction of individual parameters that changed by more than a tiny epsilon. 100% = every weight updated." />
                        </p>
                        <p className="text-lg font-mono font-semibold text-white">{(selectedLayer.percent_changed * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 mb-1">
                          Max Δ
                          <MetricHelp term="Max Delta" explanation="The largest single weight change. Very large values may indicate a gradient explosion or learning rate issue." />
                        </p>
                        <p className="text-lg font-mono font-semibold text-white">{selectedLayer.max_delta.toFixed(6)}</p>
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-4">
                    {/* Delta histogram */}
                    {selectedLayer.delta_histogram && (
                      <div className="card">
                        <h3 className="text-sm font-medium text-slate-400 mb-1">Weight Delta Distribution</h3>
                        <p className="text-xs text-slate-600 mb-3">
                          How much each weight changed. A bell curve centered at 0 is healthy. Skewed or fat tails = instability.
                        </p>
                        <ResponsiveContainer width="100%" height={180}>
                          <BarChart data={
                            selectedLayer.delta_histogram.counts.map((count, i) => ({
                              bin: ((selectedLayer.delta_histogram.bin_edges[i] + selectedLayer.delta_histogram.bin_edges[i + 1]) / 2).toFixed(4),
                              count,
                            }))
                          } margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="bin" stroke="#475569" tick={{ fontSize: 8 }} interval="preserveStartEnd" />
                            <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                            <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }} />
                            <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* Radar chart */}
                    {radarData.length > 0 && (
                      <div className="card">
                        <h3 className="text-sm font-medium text-slate-400 mb-1">Change Profile</h3>
                        <p className="text-xs text-slate-600 mb-3">
                          Bigger area = bigger overall change. Balanced shape = consistent change across all aspects.
                        </p>
                        <ResponsiveContainer width="100%" height={180}>
                          <RadarChart data={radarData}>
                            <PolarGrid stroke="#1e293b" />
                            <PolarAngleAxis dataKey="metric" tick={{ fontSize: 9, fill: '#64748b' }} />
                            <Radar name="change" dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.3} />
                          </RadarChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>

                  {/* Top-K deltas */}
                  {selectedLayer.top_k_deltas && (
                    <div className="card">
                      <h3 className="text-sm font-medium text-slate-400 mb-1">Top Changed Parameters</h3>
                      <p className="text-xs text-slate-600 mb-3">
                        The individual weights that changed the most. Concentrated large changes may indicate a specific feature detector was relearned.
                      </p>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="text-xs text-slate-500 border-b border-slate-800">
                              <th className="text-left py-2 pr-4">#</th>
                              <th className="text-left py-2 pr-4">Index</th>
                              <th className="text-right py-2 pr-4">Delta</th>
                              <th className="text-right py-2">|Delta|</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedLayer.top_k_deltas.map((delta, i) => (
                              <tr key={i} className="border-b border-slate-800/50">
                                <td className="py-1.5 pr-4 text-slate-500">{i + 1}</td>
                                <td className="py-1.5 pr-4 font-mono text-xs text-slate-400">
                                  [{delta.index?.join(', ')}]
                                </td>
                                <td className={`py-1.5 pr-4 text-right font-mono ${delta.value > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                  {delta.value > 0 ? '+' : ''}{delta.value.toFixed(6)}
                                </td>
                                <td className="py-1.5 text-right font-mono text-slate-300">
                                  {delta.abs_value.toFixed(6)}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="card text-center py-16">
                  <Layers className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">Select a layer to inspect its changes</p>
                  <p className="text-xs text-slate-600 mt-2">Each metric tells a different story about how that layer evolved</p>
                </div>
              )}
            </div>
          </div>

          {/* Alert for critical layers */}
          {diff.summary?.critical > 0 && (
            <div className="card border-red-500/30 mt-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="text-red-400 font-medium">Critical Drift Detected</h4>
                  <p className="text-sm text-slate-400 mt-1">
                    {diff.summary.critical} layer{diff.summary.critical > 1 ? 's' : ''} show critical weight changes.
                    This may indicate catastrophic forgetting, a learning rate issue, or a data distribution shift.
                    Consider comparing with intermediate checkpoints to isolate when the change happened.
                  </p>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {!diff && !loading && stepA && stepB && (
        <div className="card text-center py-16">
          <Zap className="w-12 h-12 text-slate-600 mx-auto mb-4" />
          <p className="text-slate-400">Select two checkpoints above to compare them</p>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────
// NOTE: duplicate export removed — only one DiffViewer above
// ──────────────────────────────────────────────────────────────
/* eslint-disable no-unreachable */
function _DiffViewerDuplicate() {
  const { runId } = useParams()
  const [searchParams, setSearchParams] = useSearchParams()
  const [diff, setDiff] = useState(null)
  const [checkpoints, setCheckpoints] = useState([])
  const [stepA, setStepA] = useState(searchParams.get('a') || '')
  const [stepB, setStepB] = useState(searchParams.get('b') || '')
  const [selectedLayer, setSelectedLayer] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Load checkpoints for selector
  useEffect(() => {
    fetchCheckpoints(runId)
      .then(data => {
        const ckpts = data.checkpoints || []
        setCheckpoints(ckpts)
        if (!stepA && !stepB && ckpts.length >= 2) {
          setStepA(String(ckpts[ckpts.length - 2].step))
          setStepB(String(ckpts[ckpts.length - 1].step))
        }
      })
      .catch(() => {})
  }, [runId])

  // Load diff when steps change
  useEffect(() => {
    if (stepA && stepB && stepA !== stepB) {
      setLoading(true)
      setError(null)
      fetchDiff(runId, parseInt(stepA), parseInt(stepB), true)
        .then(data => {
          setDiff(data)
          setLoading(false)
          if (data.layers?.length > 0) setSelectedLayer(data.layers[0])
        })
        .catch(err => {
          setError(err.message)
          setLoading(false)
        })
    }
  }, [runId, stepA, stepB])

  return (
    <div>
      {/* Header with checkpoint selector */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Diff Viewer</h1>
          <p className="text-sm text-slate-500 mt-1">Compare model weights between checkpoints</p>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs text-slate-500">Checkpoint A:</label>
          <select
            value={stepA}
            onChange={e => setStepA(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-glass-500"
          >
            <option value="">Select...</option>
            {checkpoints.map(c => (
              <option key={c.step} value={c.step}>
                Step {c.step?.toLocaleString()} {c.tag ? `(${c.tag})` : ''}
              </option>
            ))}
          </select>

          <span className="text-slate-600">vs</span>

          <label className="text-xs text-slate-500">Checkpoint B:</label>
          <select
            value={stepB}
            onChange={e => setStepB(e.target.value)}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-glass-500"
          >
            <option value="">Select...</option>
            {checkpoints.map(c => (
              <option key={c.step} value={c.step}>
                Step {c.step?.toLocaleString()} {c.tag ? `(${c.tag})` : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <LoadingSpinner />}
      {error && <ErrorMessage message={error} />}

      {diff && !loading && (
        <div className="grid grid-cols-12 gap-4">
          {/* Left panel: Layer tree */}
          <div className="col-span-4">
            {/* Summary */}
            <div className="card mb-4 bg-slate-900/60 border-indigo-500/20">
              <h3 className="text-sm font-semibold text-indigo-400 mb-3 uppercase tracking-wider">Impact Summary</h3>
              <div className="grid grid-cols-2 gap-3">
                <MetricValue label="Total Layers" value={diff.summary?.total_layers || 0} />
                <MetricValue label="Critical Drift" value={diff.summary?.critical || 0} className={diff.summary?.critical > 0 ? "text-red-400" : "text-slate-400"} />
                <MetricValue label="High Impact" value={diff.summary?.high || 0} className={diff.summary?.high > 0 ? "text-orange-400" : "text-slate-400"} />
                <MetricValue label="Medium Impact" value={diff.summary?.medium || 0} className={diff.summary?.medium > 0 ? "text-amber-400" : "text-slate-400"} />
              </div>
            </div>

            {/* Layer list */}
            <div className="card bg-slate-900/40">
              <h3 className="text-sm font-semibold text-slate-300 mb-3">Model Layers</h3>
              <div className="space-y-1.5 max-h-[600px] overflow-y-auto custom-scrollbar pr-2">
                {diff.layers?.map((layer, i) => (
                  <button
                    key={i}
                    onClick={() => setSelectedLayer(layer)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg transition-all text-sm border hover:-translate-y-0.5 ${
                      selectedLayer?.name === layer.name
                        ? 'bg-indigo-500/10 border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.1)]'
                        : 'bg-slate-800/50 border-slate-700/50 hover:bg-slate-800/80 hover:border-slate-600'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className={`font-mono text-xs truncate mr-2 ${selectedLayer?.name === layer.name ? 'text-indigo-300' : 'text-slate-300'}`}>{layer.name}</span>
                      <SeverityBadge severity={layer.severity} />
                    </div>
                    <div className="flex justify-between items-center mt-2 group">
                      <div className="flex gap-3 text-[11px] font-mono">
                        <span className="text-slate-400">ΔW=<span className="text-slate-300">{layer.frob_norm.toFixed(4)}</span></span>
                        <span className="text-slate-400">cos=<span className="text-slate-300">{layer.cos_sim.toFixed(4)}</span></span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right panel: Layer details */}
          <div className="col-span-8">
            {selectedLayer ? (
              <>
                {/* Layer header */}
                <div className="card mb-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="font-mono text-lg font-medium text-white">{selectedLayer.name}</h3>
                      <p className="text-xs text-slate-500 mt-1">
                        Shape: [{selectedLayer.shape?.join(', ')}]
                      </p>
                    </div>
                    <SeverityBadge severity={selectedLayer.severity} />
                  </div>

                  <div className="grid grid-cols-4 gap-4 mt-4">
                    <MetricValue label="Frobenius Norm" value={selectedLayer.frob_norm.toFixed(4)} />
                    <MetricValue label="Cosine Similarity" value={selectedLayer.cos_sim.toFixed(4)} />
                    <MetricValue label="% Changed" value={`${(selectedLayer.percent_changed * 100).toFixed(1)}`} unit="%" />
                    <MetricValue label="Max Delta" value={selectedLayer.max_delta.toFixed(6)} />
                  </div>
                </div>

                {/* Delta histogram */}
                {selectedLayer.delta_histogram && (
                  <div className="card mb-4">
                    <h3 className="text-sm font-medium text-slate-400 mb-4">Weight Delta Distribution</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={
                        selectedLayer.delta_histogram.counts.map((count, i) => ({
                          bin: ((selectedLayer.delta_histogram.bin_edges[i] + selectedLayer.delta_histogram.bin_edges[i + 1]) / 2).toFixed(4),
                          count,
                        }))
                      }>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="bin" stroke="#475569" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
                        <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                        />
                        <Bar dataKey="count" fill="#6366f1" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Top-K deltas */}
                {selectedLayer.top_k_deltas && (
                  <div className="card">
                    <h3 className="text-sm font-medium text-slate-400 mb-3">Top-K Changed Parameters</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-xs text-slate-500 border-b border-slate-800">
                            <th className="text-left py-2 pr-4">#</th>
                            <th className="text-left py-2 pr-4">Index</th>
                            <th className="text-right py-2 pr-4">Delta Value</th>
                            <th className="text-right py-2">|Delta|</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedLayer.top_k_deltas.map((delta, i) => (
                            <tr key={i} className="border-b border-slate-800/50">
                              <td className="py-1.5 pr-4 text-slate-500">{i + 1}</td>
                              <td className="py-1.5 pr-4 font-mono text-xs text-slate-400">
                                [{delta.index?.join(', ')}]
                              </td>
                              <td className={`py-1.5 pr-4 text-right font-mono ${
                                delta.value > 0 ? 'text-emerald-400' : 'text-red-400'
                              }`}>
                                {delta.value > 0 ? '+' : ''}{delta.value.toFixed(6)}
                              </td>
                              <td className="py-1.5 text-right font-mono text-slate-300">
                                {delta.abs_value.toFixed(6)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="card text-center py-16">
                <Layers className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                <p className="text-slate-400">Select a layer to view diff details</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Alert for critical layers */}
      {diff && diff.summary?.critical > 0 && (
        <div className="card border-red-500/30 mt-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <h4 className="text-red-400 font-medium">Critical Drift Detected</h4>
              <p className="text-sm text-slate-400 mt-1">
                {diff.summary.critical} layer{diff.summary.critical > 1 ? 's' : ''} show critical weight changes.
                This may indicate catastrophic forgetting, learning rate issues, or data distribution shift.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
