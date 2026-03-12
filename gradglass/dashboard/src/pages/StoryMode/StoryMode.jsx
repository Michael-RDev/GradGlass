import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchRun, fetchEvalLab, fetchMetrics, fetchCheckpoints, fetchAnalysis } from '../../api'
import { StatusBadge, LoadingSpinner, ErrorMessage } from '../../components/ui'
import { CheckCircle2, TrendingUp, AlertTriangle, ArrowRight } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const CHART_COLORS = ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#38bdf8', '#f472b6']

/** Pick the best numeric metric keys from a metrics array to chart. */
function pickChartKeys(metrics) {
  if (!metrics?.length) return []
  const sample = metrics.find(m => Object.keys(m).length > 2) || metrics[0]
  const skip = new Set(['step', 'timestamp', 'fit_duration_s'])
  const allKeys = Object.keys(sample).filter(k => !skip.has(k) && typeof sample[k] === 'number')
  // Prefer loss-like keys (neural nets) or eval-metric keys (XGBoost/LightGBM)
  const lossKeys = allKeys.filter(k =>
    k === 'loss' || k.includes('loss') || k.includes('logloss') ||
    k.includes('rmse') || k.includes('error') || k.includes('auc')
  )
  return (lossKeys.length > 0 ? lossKeys : allKeys).slice(0, 4)
}

export default function StoryMode() {
  const { runId } = useParams()
  const [data, setData] = useState({ run: null, evalLab: null, metrics: null, checkpoints: null })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([
      fetchRun(runId),
      fetchEvalLab(runId).catch(() => null),
      fetchMetrics(runId),
      fetchCheckpoints(runId),
      fetchAnalysis(runId).catch(() => null),
    ])
      .then(([run, evalLab, metrics, checkpoints, analysis]) => {
        setData({ run, evalLab, metrics: metrics.metrics, checkpoints: checkpoints.checkpoints, analysis })
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!data.run) return <ErrorMessage message="Run not found" />

  const { run, evalLab, metrics, analysis } = data
  const latestEval = evalLab?.evaluations?.[evalLab.evaluations.length - 1]
  const isBoosting = ['xgboost', 'lightgbm', 'sklearn'].includes(run.framework)

  // Status computation — checked in priority order
  let healthStatus = 'Healthy'
  let Icon = CheckCircle2
  let color = 'text-emerald-400'
  let healthReason = isBoosting
    ? 'Ensemble training completed successfully. Check Performance for accuracy / F1 scores and Architecture for feature importances.'
    : 'Training is progressing normally. Loss is decreasing and no unstable gradients detected.'

  // 1. Hard failure from run status
  if (run.status === 'failed') {
    healthStatus = 'Failed'
    Icon = AlertTriangle
    color = 'text-red-400'
    healthReason = 'Training failed or was interrupted early.'

  // 2. Any FAIL-status tests in the analysis report (highest priority signal)
  } else if (analysis?.tests?.results) {
    const failures = analysis.tests.results.filter(t => t.status === 'fail')
    const warnings = analysis.tests.results.filter(t => t.status === 'warn')
    const overfitFail = failures.find(t =>
      ['OVERFITTING_HEURISTIC', 'TRAIN_VAL_GAP', 'VAL_LOSS_DIVERGENCE'].includes(t.id)
    )
    const criticalFail = failures.find(t => t.severity === 'CRITICAL')
    if (overfitFail) {
      healthStatus = 'Overfitting'
      Icon = TrendingUp
      color = 'text-red-400'
      healthReason = overfitFail.recommendation ||
        'Validation loss is rising while training loss falls. The model has memorised the training set.'
    } else if (criticalFail || failures.length > 0) {
      healthStatus = 'Issues Detected'
      Icon = AlertTriangle
      color = 'text-orange-400'
      healthReason = `${failures.length} test${failures.length > 1 ? 's' : ''} failed. Open the Diagnostic Report for details.`
    } else {
      // Check for gradient instability warnings even without hard failures
      const GRAD_WARN_IDS = ['GRAD_LAYER_IMBALANCE', 'GRAD_VANISHING', 'GRAD_EXPLODING', 'GRAD_NAN_INF']
      const gradWarn = warnings.find(t => GRAD_WARN_IDS.includes(t.id))
      // Only use the norm-based builtin test results for the health badge.
      // gradient_flow_analysis flagged_layers uses a mean-based VANISHING check
      // that fires on converged models (signed grads cancel → mean ≈ 0) even
      // when norms are perfectly healthy — so we intentionally ignore it here.
      if (gradWarn) {
        healthStatus = 'Unstable Gradients'
        Icon = AlertTriangle
        color = 'text-orange-400'
        const flaggedCount = gfa?.flagged || 0
        healthReason = `Gradient instability detected across ${flaggedCount} layer${flaggedCount !== 1 ? 's' : ''}. Check the Gradient Flow page for details.`
      }
    }

  // 3. Fallback: check val_loss trend directly from metrics
  } else if (!isBoosting && metrics && metrics.length > 5) {
    const valLosses = metrics.filter(m => m.val_loss != null).map(m => m.val_loss)
    if (valLosses.length >= 4) {
      const half = valLosses.slice(Math.floor(valLosses.length / 2))
      const risingSteps = half.filter((v, i) => i > 0 && v > half[i - 1]).length
      if (risingSteps / Math.max(half.length - 1, 1) >= 0.8) {
        healthStatus = 'Overfitting'
        Icon = TrendingUp
        color = 'text-red-400'
        healthReason = 'Validation loss is rising consistently while training continues. Try early stopping or regularization.'
      }
    } else {
      const lastLosses = metrics.slice(-5).map(m => m.loss).filter(v => v != null)
      if (lastLosses.length === 5 && lastLosses[4] > lastLosses[0] * 1.5) {
        healthStatus = 'Unstable'
        Icon = TrendingUp
        color = 'text-orange-400'
        healthReason = 'Loss spikes detected in recent steps. Model might be diverging.'
      }
    }
  }

  const chartKeys = pickChartKeys(metrics)
  const downsampledMetrics = metrics
    ? metrics.filter((_, i) => i % Math.ceil(metrics.length / 150) === 0)
    : []

  return (
    <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="mb-8">
        <h1 className="text-4xl font-bold tracking-tight text-white mb-3">
          {run.name}
        </h1>
        <div className="flex items-center gap-4 text-sm">
          <StatusBadge status={run.status} />
          {run.framework && (
            <span className="text-xs text-slate-500 bg-slate-800 px-2 py-0.5 rounded-md font-mono">{run.framework}</span>
          )}
          <span className="text-slate-500 font-mono">{runId}</span>
          <span className="text-slate-500">{run.start_time}</span>
        </div>
      </div>

      <div className="space-y-6">
        {/* Card A: Status at a Glance */}
        <section className="card bg-slate-900/40 relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-indigo-500 to-violet-500 opacity-50"></div>
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            Status at a Glance
          </h2>

          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <Icon className={`w-8 h-8 ${color}`} />
                <span className={`text-3xl font-bold ${color}`}>{healthStatus}</span>
              </div>
              <p className="text-slate-400 max-w-lg mt-2 leading-relaxed">
                {healthReason}
              </p>
            </div>

            <div className="flex flex-col gap-2">
              <Link to={`/run/${runId}/analysis`} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 transition-colors rounded-lg text-sm font-medium text-slate-200">
                View Diagnostic Report
              </Link>
              <Link to={`/run/${runId}/architecture`} className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 transition-colors rounded-lg text-sm font-medium text-slate-200">
                {isBoosting ? '🌲 View Trees & Importances' : 'View Architecture'}
              </Link>
            </div>
          </div>
        </section>

        {/* Card B: Performance */}
        <section className="card bg-slate-900/40 relative overflow-hidden">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold">Performance</h2>
            {latestEval && (
              <Link to={`/run/${runId}/eval`} className="text-indigo-400 text-sm hover:text-indigo-300 flex items-center gap-1 group">
                Open Eval Lab <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Link>
            )}
          </div>

          {!latestEval ? (
            <div className="text-center py-8">
              <p className="text-slate-500 mb-2">No evaluation data available.</p>
              <p className="text-sm text-slate-600 font-mono">Call `run.log_batch()` to auto-generate performance metrics.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {latestEval.is_classification ? (
                <>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-slate-400 text-sm mb-1">Macro F1</div>
                    <div className="text-2xl font-bold font-mono text-indigo-400">
                      {(latestEval.macro_f1 * 100).toFixed(1)}<span className="text-sm ml-1 text-slate-500">%</span>
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-slate-400 text-sm mb-1">Accuracy</div>
                    <div className="text-2xl font-bold font-mono text-emerald-400">
                      {(latestEval.accuracy * 100).toFixed(1)}<span className="text-sm ml-1 text-slate-500">%</span>
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-slate-400 text-sm mb-1">Classes Tracked</div>
                    <div className="text-2xl font-bold font-mono text-slate-200">
                      {latestEval.confusion_matrix?.classes?.length || 0}
                    </div>
                  </div>
                </>
              ) : (
                <>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-slate-400 text-sm mb-1">RMSE</div>
                    <div className="text-2xl font-bold font-mono text-orange-400">
                      {latestEval.rmse?.toFixed(4) ?? '—'}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-slate-400 text-sm mb-1">MAE</div>
                    <div className="text-2xl font-bold font-mono text-indigo-400">
                      {latestEval.mae?.toFixed(4) ?? '—'}
                    </div>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Dynamic training metrics chart */}
          {downsampledMetrics.length > 0 && chartKeys.length > 0 && (
            <div className="mt-8">
              <h3 className="text-sm font-medium text-slate-400 mb-4 uppercase tracking-wider">
                {chartKeys.includes('loss') ? 'Training Loss' : 'Training Metrics'}
              </h3>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={downsampledMetrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 11 }} tickLine={false} />
                    <YAxis stroke="#475569" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    {chartKeys.length > 1 && <Legend wrapperStyle={{ fontSize: 11, color: '#94a3b8' }} />}
                    {chartKeys.map((key, idx) => (
                      <Line
                        key={key}
                        type="monotone"
                        dataKey={key}
                        stroke={CHART_COLORS[idx % CHART_COLORS.length]}
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                        name={key}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </section>

        {/* Card D: Likely Causes */}
        <section className="card bg-slate-900/40 relative overflow-hidden">
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            Likely Causes
          </h2>
          <div className="text-center py-8">
            <p className="text-slate-500 mb-4">No significant anomalies detected to trace yet.</p>
            <Link to={`/run/${runId}/rootcause`} className="text-indigo-400 hover:text-indigo-300 text-sm transition-colors">
              Explore the Root Cause Map
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}
