import { useState, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchAnalysis, fetchFreezeCode } from '../api'
import { LoadingSpinner, ErrorMessage, SeverityBadge } from '../components/ui'
import {
  CheckCircle, AlertTriangle, XCircle, SkipForward, Search,
  ChevronDown, ChevronRight, Database, Cpu, TrendingDown,
  GitCompare, Activity, Layers, BarChart3, Globe, RefreshCw,
  Shield, FileText, Snowflake, Copy, Code2, Zap, Eye
} from 'lucide-react'

const STATUS_CONFIG = {
  pass: { icon: CheckCircle, color: 'text-emerald-400', bg: 'bg-emerald-400/10', border: 'border-emerald-400/30', label: 'Pass' },
  warn: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-400/10', border: 'border-amber-400/30', label: 'Warn' },
  fail: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-400/10', border: 'border-red-400/30', label: 'Fail' },
  skip: { icon: SkipForward, color: 'text-slate-500', bg: 'bg-slate-500/10', border: 'border-slate-500/30', label: 'Skip' },
}

const SEVERITY_COLORS = {
  LOW: 'text-slate-400 bg-slate-400/10',
  MEDIUM: 'text-yellow-400 bg-yellow-400/10',
  HIGH: 'text-orange-400 bg-orange-400/10',
  CRITICAL: 'text-red-400 bg-red-400/10',
}

const CATEGORY_ICONS = {
  'Artifact & Store Integrity': Database,
  'Model Structure': Cpu,
  'Training Metrics': TrendingDown,
  'Checkpoint Diff': GitCompare,
  'Gradient Flow': Activity,
  'Activations': Layers,
  'Predictions': BarChart3,
  'Data': FileText,
  'Distributed Training': Globe,
  'Reproducibility': RefreshCw,
}

const CATEGORY_DESCRIPTIONS = {
  'Artifact & Store Integrity': 'Validates checkpoint files, metadata, and store layout',
  'Model Structure': 'Checks architecture serialization, DAG validity, and layer config',
  'Training Metrics': 'Analyzes loss curves, accuracy, LR tracking, and epoch-level improvements',
  'Checkpoint Diff': 'Compares weight changes, severity distribution, and dead layers',
  'Gradient Flow': 'Detects vanishing/exploding gradients, saliency, and freeze candidates',
  'Activations': 'Checks for NaN, sparsity collapse, dead channels, saturation, and capacity utilization',
  'Predictions': 'Tracks label flips, confidence shifts, LIME-proxy variance, and prediction stability',
  'Data': 'Validates dataset hashes, class balance, and slice coverage',
  'Distributed Training': 'Monitors rank health, sync latency, and straggler detection',
  'Reproducibility': 'Verifies seed, environment, determinism flags, and git commit',
}

// ──────────────────────────────────────────────────────────────
// Freeze Layer Weights Panel
// ──────────────────────────────────────────────────────────────
function FreezePanel({ runId }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [framework, setFramework] = useState('pytorch')
  const [copied, setCopied] = useState(false)
  const [mode, setMode] = useState(null) // null | 'freeze' | 'nothing'

  const load = () => {
    setLoading(true)
    setError(null)
    fetchFreezeCode(runId)
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }

  const code = data ? (framework === 'pytorch' ? data.pytorch_code : data.tensorflow_code) : ''

  const copyCode = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="card mb-6 border-blue-500/20 bg-slate-900/50">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Snowflake className="w-4 h-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-blue-300">Freeze Layer Weights</h3>
          <span className="text-xs text-slate-600">Edit Python Source Code</span>
        </div>
        {!data && !loading && (
          <button onClick={load}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-blue-600/20 text-blue-300 text-xs hover:bg-blue-600/30 transition-colors">
            <Zap className="w-3.5 h-3.5" /> Analyze Layers
          </button>
        )}
      </div>

      {loading && <div className="text-xs text-slate-500 animate-pulse">Analyzing gradient activity…</div>}
      {error && <div className="text-xs text-red-400">{error}</div>}

      {data && (
        <>
          <p className="text-sm text-slate-300 mb-4">{data.message}</p>

          {data.candidates?.length > 0 && (
            <div className="mb-4">
              <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider">Freeze Candidates (low gradient activity)</p>
              <div className="space-y-1">
                {data.candidates.map((c, i) => (
                  <div key={i} className="flex items-center justify-between px-3 py-1.5 bg-slate-800/60 rounded text-xs font-mono">
                    <span className="text-blue-300">{c.layer}</span>
                    <div className="flex items-center gap-3 text-slate-500">
                      <span>∇norm = {c.mean_grad_norm.toFixed(2e-8 > 0 ? 8 : 4)}</span>
                      <span className="text-slate-600">{(c.relative_norm * 100).toFixed(2)}% of max</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Action choice */}
          {mode === null && (
            <div className="flex gap-3 mt-4">
              <button onClick={() => setMode('freeze')}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600/20 text-blue-300 text-sm hover:bg-blue-600/30 border border-blue-500/30 transition-colors">
                <Code2 className="w-4 h-4" /> Show Freeze Code
              </button>
              <button onClick={() => setMode('nothing')}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 text-slate-400 text-sm hover:bg-slate-700 border border-slate-700 transition-colors">
                Do Nothing
              </button>
            </div>
          )}

          {mode === 'nothing' && (
            <div className="mt-4 p-3 bg-slate-800/40 rounded-lg border border-slate-700/40">
              <p className="text-sm text-slate-400">✓ Keeping all layers trainable — no source code changes needed.</p>
              <button onClick={() => setMode(null)} className="text-xs text-slate-600 hover:text-slate-400 mt-2 transition-colors">← Back</button>
            </div>
          )}

          {mode === 'freeze' && (
            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex gap-2">
                  {['pytorch', 'tensorflow'].map(f => (
                    <button key={f} onClick={() => setFramework(f)}
                      className={`text-xs px-3 py-1 rounded-lg font-mono transition-colors ${framework === f ? 'bg-glass-600/30 text-glass-300' : 'text-slate-500 hover:text-slate-300'}`}>
                      {f === 'pytorch' ? 'PyTorch' : 'TensorFlow/Keras'}
                    </button>
                  ))}
                </div>
                <button onClick={copyCode}
                  className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition-colors">
                  <Copy className="w-3 h-3" />
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
              <pre className="text-xs text-slate-300 bg-slate-900 border border-slate-700 rounded-lg p-4 overflow-x-auto max-h-80 overflow-y-auto font-mono leading-relaxed">
                {code}
              </pre>
              <p className="text-xs text-slate-600 mt-2">
                Copy this into your training script. Uncomment lines for layers you want to freeze.
              </p>
              <button onClick={() => setMode(null)} className="text-xs text-slate-600 hover:text-slate-400 mt-2 transition-colors">← Back</button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function StatusIcon({ status, size = 'w-4 h-4' }) {
  const cfg = STATUS_CONFIG[status] || STATUS_CONFIG.skip
  const Icon = cfg.icon
  return <Icon className={`${size} ${cfg.color}`} />
}

function TestCard({ test, isExpanded, onToggle }) {
  const cfg = STATUS_CONFIG[test.status] || STATUS_CONFIG.skip
  const Icon = cfg.icon

  return (
    <div className={`border rounded-lg ${cfg.border} ${cfg.bg} overflow-hidden transition-all`}>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors"
      >
        <Icon className={`w-5 h-5 ${cfg.color} shrink-0`} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-slate-200 truncate">{test.title}</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${SEVERITY_COLORS[test.severity] || ''}`}>
              {test.severity}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
              {test.category}
            </span>
          </div>
          <span className="text-xs text-slate-500 font-mono">{test.id}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {test.duration_ms > 0 && (
            <span className="text-xs text-slate-600">{test.duration_ms.toFixed(0)}ms</span>
          )}
          {isExpanded ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-3 border-t border-white/5">
          {test.recommendation && (
            <div className="mt-2 flex items-start gap-2">
              <Shield className="w-4 h-4 text-glass-400 mt-0.5 shrink-0" />
              <p className="text-sm text-glass-300">{test.recommendation}</p>
            </div>
          )}
          {test.details && Object.keys(test.details).length > 0 && (
            <div className="mt-2">
              <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-2 overflow-x-auto max-h-48 overflow-y-auto">
                {JSON.stringify(test.details, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Analysis() {
  const { runId } = useParams()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [categoryFilter, setCategoryFilter] = useState('all')
  const [severityFilter, setSeverityFilter] = useState('all')
  const [expandedTests, setExpandedTests] = useState(new Set())

  useEffect(() => {
    fetchAnalysis(runId)
      .then(data => {
        setReport(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  const toggleTest = (testId) => {
    setExpandedTests(prev => {
      const next = new Set(prev)
      if (next.has(testId)) next.delete(testId)
      else next.add(testId)
      return next
    })
  }

  const testResults = report?.tests?.results || []
  const categories = useMemo(() => [...new Set(testResults.map(t => t.category))].sort(), [testResults])

  const filteredTests = useMemo(() => {
    return testResults.filter(t => {
      if (statusFilter !== 'all' && t.status !== statusFilter) return false
      if (categoryFilter !== 'all' && t.category !== categoryFilter) return false
      if (severityFilter !== 'all' && t.severity !== severityFilter) return false
      if (searchQuery) {
        const q = searchQuery.toLowerCase()
        return t.id.toLowerCase().includes(q) ||
               t.title.toLowerCase().includes(q) ||
               t.category.toLowerCase().includes(q)
      }
      return true
    })
  }, [testResults, statusFilter, categoryFilter, severityFilter, searchQuery])

  const groupedTests = useMemo(() => {
    const groups = {}
    filteredTests.forEach(t => {
      if (!groups[t.category]) groups[t.category] = []
      groups[t.category].push(t)
    })
    return groups
  }, [filteredTests])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!report) return <ErrorMessage message="No analysis report found" />

  const ckpt = report.checkpoint_diff_summary || {}
  const grad = report.gradient_flow_analysis || {}
  const met = report.training_metrics_summary || {}
  const store = report.artifact_store_summary || {}
  const tests = report.tests || {}

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Post-Run Analysis</h1>
          <p className="text-sm text-slate-500 mt-1">
            Generated {report.generated_at} • {testResults.length} tests
          </p>
        </div>
      </div>

      {/* Test Suite Overview */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        <button onClick={() => setStatusFilter('all')}
          className={`card text-center cursor-pointer transition-all ${statusFilter === 'all' ? 'ring-1 ring-glass-500' : ''}`}>
          <div className="text-2xl font-bold">{tests.total || 0}</div>
          <div className="text-xs text-slate-500 uppercase tracking-wider">Total</div>
        </button>
        <button onClick={() => setStatusFilter(statusFilter === 'pass' ? 'all' : 'pass')}
          className={`card text-center cursor-pointer transition-all ${statusFilter === 'pass' ? 'ring-1 ring-emerald-500' : ''}`}>
          <div className="text-2xl font-bold text-emerald-400">{tests.passed || 0}</div>
          <div className="text-xs text-emerald-400/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <CheckCircle className="w-3 h-3" /> Pass
          </div>
        </button>
        <button onClick={() => setStatusFilter(statusFilter === 'warn' ? 'all' : 'warn')}
          className={`card text-center cursor-pointer transition-all ${statusFilter === 'warn' ? 'ring-1 ring-amber-500' : ''}`}>
          <div className="text-2xl font-bold text-amber-400">{tests.warned || 0}</div>
          <div className="text-xs text-amber-400/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <AlertTriangle className="w-3 h-3" /> Warn
          </div>
        </button>
        <button onClick={() => setStatusFilter(statusFilter === 'fail' ? 'all' : 'fail')}
          className={`card text-center cursor-pointer transition-all ${statusFilter === 'fail' ? 'ring-1 ring-red-500' : ''}`}>
          <div className="text-2xl font-bold text-red-400">{tests.failed || 0}</div>
          <div className="text-xs text-red-400/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <XCircle className="w-3 h-3" /> Fail
          </div>
        </button>
        <button onClick={() => setStatusFilter(statusFilter === 'skip' ? 'all' : 'skip')}
          className={`card text-center cursor-pointer transition-all ${statusFilter === 'skip' ? 'ring-1 ring-slate-500' : ''}`}>
          <div className="text-2xl font-bold text-slate-500">{tests.skipped || 0}</div>
          <div className="text-xs text-slate-500/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <SkipForward className="w-3 h-3" /> Skip
          </div>
        </button>
      </div>

      {/* Summary Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">        {/* Checkpoint Diff Summary */}
        <div className="card">
          <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
            <GitCompare className="w-4 h-4 text-glass-500" />
            Checkpoint Diff Summary
          </h3>
          <div className="text-sm text-slate-300 space-y-1">
            <p>Checkpoints saved: <span className="font-mono font-medium">{ckpt.checkpoints_saved || 0}</span></p>
            {ckpt.checkpoints?.map((c, i) => (
              <p key={i} className="text-xs text-slate-500 font-mono pl-4">
                step={c.step} {c.tag && `[${c.tag}]`} params={c.num_params?.toLocaleString()} size={c.size_mb} MB
              </p>
            ))}
            {ckpt.diff && (
              <div className="mt-3 pt-3 border-t border-slate-800">
                <p className="text-xs text-slate-400">
                  Diff: step {ckpt.diff.step_a} → step {ckpt.diff.step_b} • {ckpt.diff.layers_compared} layers
                </p>
                <div className="flex gap-3 mt-1 text-xs font-mono">
                  {Object.entries(ckpt.diff.severity_counts || {}).map(([k, v]) => (
                    <span key={k} className={`${k === 'critical' ? 'text-red-400' : k === 'high' ? 'text-orange-400' : k === 'medium' ? 'text-yellow-400' : 'text-slate-500'}`}>
                      {k}: {v}
                    </span>
                  ))}
                </div>
                {ckpt.diff.top_changed_layers?.length > 0 && (
                  <div className="mt-2 space-y-0.5">
                    <p className="text-xs text-slate-500">Top changed:</p>
                    {ckpt.diff.top_changed_layers.map((lr, i) => (
                      <p key={i} className="text-xs font-mono text-slate-400 pl-2">
                        {lr.layer} <span className="text-slate-600">frob={lr.frob_norm}</span> <SeverityBadge severity={lr.severity} />
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Training Metrics Summary */}
        <div className="card">
          <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-glass-500" />
            Training Metrics Summary
          </h3>
          {met.has_data ? (
            <div className="text-sm text-slate-300 space-y-1">
              <p>Total steps: <span className="font-mono font-medium">{met.total_steps?.toLocaleString()}</span></p>
              {met.loss_final != null && (
                <p>Loss: <span className="font-mono">{met.loss_start}</span> → <span className="font-mono font-medium text-emerald-400">{met.loss_final}</span></p>
              )}
              {met.acc_final != null && (
                <p>Accuracy: <span className="font-mono">{met.acc_start}%</span> → <span className="font-mono font-medium text-blue-400">{met.acc_final}%</span></p>
              )}
              {met.lr_final != null && (
                <p>Learning rate: <span className="font-mono">{met.lr_final}</span></p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500 italic">No metrics data found</p>
          )}
        </div>

        {/* Gradient Flow */}
        <div className="card">
          <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
            <Activity className="w-4 h-4 text-glass-500" />
            Gradient Flow Analysis
          </h3>
          {grad.has_data ? (
            <div className="text-sm text-slate-300 space-y-1">
              <p>Layers tracked: <span className="font-mono">{grad.layers_tracked}</span></p>
              <p>Healthy: <span className="font-mono text-emerald-400">{grad.healthy}</span></p>
              <p>Flagged: <span className="font-mono text-amber-400">{grad.flagged}</span></p>
              {grad.flagged_layers?.length > 0 && (
                <div className="mt-2 space-y-0.5">
                  {grad.flagged_layers.map((fl, i) => (
                    <p key={i} className="text-xs font-mono text-amber-400/80 pl-2">
                      {fl.layer}: {fl.flags.join(', ')}
                    </p>
                  ))}
                </div>
              )}
              {!grad.flagged_layers?.length && (
                <p className="text-xs text-emerald-400 mt-1">✓ No gradient issues detected</p>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500 italic">No gradient summaries found</p>
          )}
        </div>

        {/* Artifact Store */}
        <div className="card">
          <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
            <Database className="w-4 h-4 text-glass-500" />
            Artifact Store
          </h3>
          <div className="text-sm text-slate-300 space-y-1">
            <p>Run ID: <span className="font-mono text-xs">{store.run_id}</span></p>
            <p>Storage: <span className="font-mono font-medium">{store.storage_mb} MB</span></p>
          </div>
        </div>
      </div>

      {/* ❄️ Freeze Layer Weights Panel */}
      <FreezePanel runId={runId} />

      {/* Test Suite */}
      <div className="card mb-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-400">Test Suite</h3>
          <div className="flex items-center gap-2">
            {/* Search */}
            <div className="relative">
              <Search className="w-4 h-4 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search tests..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="bg-slate-800 text-sm text-slate-300 rounded-lg pl-8 pr-3 py-1.5 border border-slate-700 focus:border-glass-500 focus:outline-none w-48"
              />
            </div>
            {/* Category filter */}
            <select
              value={categoryFilter}
              onChange={e => setCategoryFilter(e.target.value)}
              className="bg-slate-800 text-sm text-slate-300 rounded-lg px-3 py-1.5 border border-slate-700 focus:border-glass-500 focus:outline-none"
            >
              <option value="all">All Categories</option>
              {categories.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
            {/* Severity filter */}
            <select
              value={severityFilter}
              onChange={e => setSeverityFilter(e.target.value)}
              className="bg-slate-800 text-sm text-slate-300 rounded-lg px-3 py-1.5 border border-slate-700 focus:border-glass-500 focus:outline-none"
            >
              <option value="all">All Severities</option>
              <option value="CRITICAL">Critical</option>
              <option value="HIGH">High</option>
              <option value="MEDIUM">Medium</option>
              <option value="LOW">Low</option>
            </select>
          </div>
        </div>

        <p className="text-xs text-slate-500 mb-3">
          Showing {filteredTests.length} of {testResults.length} tests
        </p>

        {/* Grouped test list */}
        <div className="space-y-4">
          {Object.entries(groupedTests).sort().map(([category, tests]) => {
            const CatIcon = CATEGORY_ICONS[category] || Database
            const catDesc = CATEGORY_DESCRIPTIONS[category] || ''
            const passCount = tests.filter(t => t.status === 'pass').length
            const warnCount = tests.filter(t => t.status === 'warn').length
            const failCount = tests.filter(t => t.status === 'fail').length

            return (
              <div key={category}>
                <div className="flex items-center gap-2 mb-2">
                  <CatIcon className="w-4 h-4 text-slate-500" />
                  <h4 className="text-sm font-medium text-slate-300">{category}</h4>
                  <span className="text-xs text-slate-600">({tests.length})</span>
                  <div className="flex items-center gap-1.5 ml-auto text-xs">
                    {passCount > 0 && <span className="text-emerald-400">{passCount}✓</span>}
                    {warnCount > 0 && <span className="text-amber-400">{warnCount}⚠</span>}
                    {failCount > 0 && <span className="text-red-400">{failCount}✗</span>}
                  </div>
                </div>
                {catDesc && <p className="text-xs text-slate-600 mb-2 ml-6">{catDesc}</p>}
                <div className="space-y-1.5 ml-6">
                  {tests.map(t => (
                    <TestCard
                      key={t.id}
                      test={t}
                      isExpanded={expandedTests.has(t.id)}
                      onToggle={() => toggleTest(t.id)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
