import { useState, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchRun, fetchMetrics, fetchCheckpoints } from '../api'
import { StatusBadge, MetricValue, LoadingSpinner, ErrorMessage } from '../components/ui'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ReferenceLine } from 'recharts'
import { GitCompare, Activity, Layers, Shield, TrendingDown, Info } from 'lucide-react'

export default function RunOverview() {
  const { runId } = useParams()
  const [run, setRun] = useState(null)
  const [metrics, setMetrics] = useState([])
  const [checkpoints, setCheckpoints] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      fetchRun(runId),
      fetchMetrics(runId),
      fetchCheckpoints(runId),
    ])
      .then(([runData, metricsData, ckptData]) => {
        setRun(runData)
        setMetrics(metricsData.metrics || [])
        setCheckpoints(ckptData.checkpoints || [])
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!run) return <ErrorMessage message="Run not found" />

  // Downsample metrics for chart if too many points
  const chartMetrics = metrics.length > 500
    ? metrics.filter((_, i) => i % Math.ceil(metrics.length / 500) === 0)
    : metrics

  // Auto-detect epoch field and group by epoch for a cleaner X-axis
  const hasEpoch = chartMetrics.some(m => m.epoch != null)

  const epochData = useMemo(() => {
    if (!hasEpoch) return chartMetrics
    const groups = {}
    chartMetrics.forEach(m => {
      const e = m.epoch
      if (e == null) return
      if (!groups[e]) groups[e] = { epoch: e, loss_sum: 0, loss_n: 0, acc_sum: 0, acc_n: 0, lr: null, step: m.step }
      if (m.loss != null) { groups[e].loss_sum += m.loss; groups[e].loss_n++ }
      if (m.acc != null) { groups[e].acc_sum += m.acc; groups[e].acc_n++ }
      if (m.lr != null) groups[e].lr = m.lr
      if (m.step != null) groups[e].step = m.step
    })
    return Object.values(groups)
      .sort((a, b) => a.epoch - b.epoch)
      .map(e => ({
        epoch: e.epoch,
        step: e.step,
        loss: e.loss_n > 0 ? +(e.loss_sum / e.loss_n).toFixed(6) : null,
        acc: e.acc_n > 0 ? +(e.acc_sum / e.acc_n).toFixed(6) : null,
        lr: e.lr,
      }))
  }, [chartMetrics, hasEpoch])

  const plotData = hasEpoch ? epochData : chartMetrics
  const xKey = hasEpoch ? 'epoch' : 'step'
  const xLabel = hasEpoch ? 'Epoch' : 'Step'

  const hasLoss = plotData.some(m => m.loss != null)
  const hasAcc = plotData.some(m => m.acc != null)
  const hasLR = plotData.some(m => m.lr != null)

  // Compute loss trend direction for annotation
  const lossTrend = useMemo(() => {
    const losses = plotData.filter(m => m.loss != null).map(m => m.loss)
    if (losses.length < 2) return null
    const first = losses.slice(0, Math.ceil(losses.length / 5)).reduce((a, b) => a + b, 0) / Math.ceil(losses.length / 5)
    const last = losses.slice(-Math.ceil(losses.length / 5)).reduce((a, b) => a + b, 0) / Math.ceil(losses.length / 5)
    const pct = ((last - first) / first * 100).toFixed(1)
    return { first: first.toFixed(4), last: last.toFixed(4), pct, improving: last < first }
  }, [plotData])

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="h2 text-theme-text-primary">{run.name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <StatusBadge status={run.status} />
            {run.framework && (
              <span className="text-xs text-theme-text-secondary bg-theme-bg border border-theme-border px-2 py-0.5 rounded-md font-mono">
                {run.framework}
              </span>
            )}
            <span className="text-xs text-theme-text-secondary">{run.start_time}</span>
          </div>
        </div>

        <div className="flex gap-2">
          <Link to={`/run/${encodeURIComponent(runId)}/diff`}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-theme-primary/10 text-theme-primary text-sm hover:bg-theme-primary/20 transition-colors">
            <GitCompare className="w-4 h-4" /> Diff Viewer
          </Link>
          <Link to={`/run/${encodeURIComponent(runId)}/gradients`}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-theme-surface-hover border border-theme-border text-theme-text-primary text-sm hover:bg-theme-bg transition-colors">
            <Activity className="w-4 h-4" /> Gradients
          </Link>
          <Link to={`/run/${encodeURIComponent(runId)}/leakage`}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-theme-surface-hover border border-theme-border text-theme-text-primary text-sm hover:bg-theme-bg transition-colors">
            <Shield className="w-4 h-4" /> Leakage
          </Link>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="card">
          <MetricValue label="Total Steps" value={run.metrics_summary?.step?.toLocaleString() || '0'} />
        </div>
        <div className="card">
          <MetricValue label="Checkpoints" value={checkpoints.length} />
        </div>
        <div className="card">
          <MetricValue label="Latest Loss" value={run.metrics_summary?.loss?.toFixed(4) || '—'} />
        </div>
        <div className="card">
          <MetricValue label="Latest Acc" value={run.metrics_summary?.acc != null ? `${(run.metrics_summary.acc * 100).toFixed(1)}` : '—'} unit="%" />
        </div>
      </div>

      {/* Metric charts */}
      {plotData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
          {hasLoss && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-theme-text-primary">Loss <span className="text-xs text-theme-text-secondary">per {xLabel}</span></h3>
                {lossTrend && (
                  <span className={`text-xs px-2 py-0.5 rounded-full font-mono ${lossTrend.improving ? 'text-emerald-400 bg-emerald-400/10' : 'text-red-400 bg-red-400/10'}`}>
                    {lossTrend.improving ? '↓' : '↑'} {Math.abs(lossTrend.pct)}% ({lossTrend.first} → {lossTrend.last})
                  </span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={plotData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey={xKey} stroke="#475569" tick={{ fontSize: 11 }} label={{ value: xLabel, position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} height={36} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    labelFormatter={v => `${xLabel} ${v}`}
                  />
                  <Line type="monotone" dataKey="loss" stroke="#6366f1" strokeWidth={2} dot={hasEpoch} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {hasAcc && (
            <div className="card">
              <h3 className="text-sm font-medium text-theme-text-primary mb-4">Accuracy <span className="text-xs text-theme-text-secondary">per {xLabel}</span></h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={plotData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey={xKey} stroke="#475569" tick={{ fontSize: 11 }} label={{ value: xLabel, position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} height={36} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    labelFormatter={v => `${xLabel} ${v}`}
                    formatter={v => [`${(v*100).toFixed(2)}%`, 'Accuracy']}
                  />
                  <Line type="monotone" dataKey="acc" stroke="#10b981" strokeWidth={2} dot={hasEpoch} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {hasLR && (
            <div className="card">
              <h3 className="text-sm font-medium text-theme-text-primary mb-4">Learning Rate <span className="text-xs text-theme-text-secondary">per {xLabel}</span></h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={plotData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey={xKey} stroke="#475569" tick={{ fontSize: 11 }} label={{ value: xLabel, position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }} height={36} />
                  <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    labelStyle={{ color: '#94a3b8' }}
                    labelFormatter={v => `${xLabel} ${v}`}
                  />
                  <Line type="monotone" dataKey="lr" stroke="#f59e0b" strokeWidth={2} dot={hasEpoch} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      {/* Checkpoint timeline */}
      {checkpoints.length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium text-theme-text-primary mb-4">Checkpoint Timeline</h3>
          <div className="flex gap-2 flex-wrap">
            {checkpoints.map((ckpt, i) => (
              <div key={i} className="px-3 py-2 bg-theme-surface-hover border border-theme-border rounded-lg text-xs">
                <div className="font-mono text-theme-text-primary">Step {ckpt.step?.toLocaleString()}</div>
                {ckpt.tag && <div className="text-theme-primary mt-0.5">{ckpt.tag}</div>}
                <div className="text-theme-text-secondary mt-0.5">{ckpt.size_mb || '?'} MB</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
