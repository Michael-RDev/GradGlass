import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { fetchLeakageReport } from '../api'
import { LoadingSpinner, ErrorMessage } from '../components/ui'
import {
  CheckCircle, XCircle, Shield, AlertTriangle,
  ChevronDown, ChevronRight, Search, Database,
  BarChart3, Fingerprint, Copy, Layers, TrendingUp
} from 'lucide-react'

const SEVERITY_COLORS = {
  CRITICAL: { text: 'text-red-400', bg: 'bg-red-400/10', border: 'border-red-400/30' },
  HIGH: { text: 'text-orange-400', bg: 'bg-orange-400/10', border: 'border-orange-400/30' },
  MEDIUM: { text: 'text-yellow-400', bg: 'bg-yellow-400/10', border: 'border-yellow-400/30' },
  LOW: { text: 'text-slate-400', bg: 'bg-slate-400/10', border: 'border-slate-400/30' },
}

const CHECK_ICONS = {
  EXACT_OVERLAP: Copy,
  TRAIN_DUPLICATES: Database,
  TEST_DUPLICATES: Database,
  NEAR_DUPLICATES: Fingerprint,
  LABEL_DISTRIBUTION: BarChart3,
  FEATURE_STATS: TrendingUp,
  TARGET_CORRELATION: Layers,
}

const CHECK_DESCRIPTIONS = {
  EXACT_OVERLAP: 'Detects identical samples shared between the training and test sets, which directly inflates evaluation metrics.',
  TRAIN_DUPLICATES: 'Finds duplicate samples within the training set that may bias model learning toward repeated examples.',
  TEST_DUPLICATES: 'Finds duplicate samples within the test set that may distort evaluation metrics.',
  NEAR_DUPLICATES: 'Identifies samples that are nearly identical (high cosine similarity) between train and test splits.',
  LABEL_DISTRIBUTION: 'Compares the class label distributions between train and test to ensure they are drawn from similar populations.',
  FEATURE_STATS: 'Compares global feature statistics (mean, std) between splits to detect preprocessing differences or data drift.',
  TARGET_CORRELATION: 'Checks for features with suspiciously high correlation to the target, which may indicate label leakage.',
}

function DistributionBar({ label, trainPct, testPct }) {
  const maxPct = Math.max(trainPct, testPct, 0.01)
  const scale = 100 / maxPct
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-6 text-right text-slate-400 font-mono">{label}</span>
      <div className="flex-1 flex flex-col gap-0.5">
        <div className="flex items-center gap-1">
          <div
            className="h-2 bg-blue-500/60 rounded-sm"
            style={{ width: `${trainPct * scale}%` }}
          />
          <span className="text-slate-500">{(trainPct * 100).toFixed(1)}%</span>
        </div>
        <div className="flex items-center gap-1">
          <div
            className="h-2 bg-emerald-500/60 rounded-sm"
            style={{ width: `${testPct * scale}%` }}
          />
          <span className="text-slate-500">{(testPct * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}

function CheckCard({ check, isExpanded, onToggle }) {
  const passed = check.passed
  const sevColors = SEVERITY_COLORS[check.severity] || SEVERITY_COLORS.LOW
  const Icon = CHECK_ICONS[check.check_id] || Shield
  const description = CHECK_DESCRIPTIONS[check.check_id] || ''

  return (
    <div className={`border rounded-lg overflow-hidden transition-all ${
      passed ? 'border-emerald-400/20 bg-emerald-400/5' : `${sevColors.border} ${sevColors.bg}`
    }`}>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-white/5 transition-colors"
      >
        {passed
          ? <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />
          : <XCircle className={`w-5 h-5 ${sevColors.text} shrink-0`} />
        }
        <Icon className="w-4 h-4 text-slate-500 shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-slate-200 truncate">{check.title}</span>
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${sevColors.text} ${sevColors.bg}`}>
              {check.severity}
            </span>
          </div>
          <span className="text-xs text-slate-500 font-mono">{check.check_id}</span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {check.duration_ms > 0 && (
            <span className="text-xs text-slate-600">{check.duration_ms.toFixed(0)}ms</span>
          )}
          {isExpanded ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
        </div>
      </button>

      {isExpanded && (
        <div className="px-4 pb-4 border-t border-white/5 space-y-3">
          {description && (
            <p className="text-xs text-slate-400 mt-2">{description}</p>
          )}

          {!passed && check.recommendation && (
            <div className="flex items-start gap-2 mt-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5 shrink-0" />
              <p className="text-sm text-amber-300">{check.recommendation}</p>
            </div>
          )}

          {/* Label Distribution visualization */}
          {check.check_id === 'LABEL_DISTRIBUTION' && check.details.train_distribution && (
            <div className="mt-2">
              <p className="text-xs text-slate-500 mb-2">
                <span className="inline-block w-3 h-2 bg-blue-500/60 rounded-sm mr-1" /> Train
                <span className="inline-block w-3 h-2 bg-emerald-500/60 rounded-sm ml-3 mr-1" /> Test
              </p>
              <div className="space-y-1.5">
                {Object.keys(check.details.train_distribution).sort().map(label => (
                  <DistributionBar
                    key={label}
                    label={label}
                    trainPct={check.details.train_distribution[label]}
                    testPct={check.details.test_distribution?.[label] || 0}
                  />
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-2">
                Max class difference: <span className="font-mono text-slate-300">
                  {(check.details.max_absolute_diff * 100).toFixed(2)}%
                </span>
              </p>
            </div>
          )}

          {/* Feature Stats visualization */}
          {check.check_id === 'FEATURE_STATS' && (
            <div className="grid grid-cols-2 gap-3 mt-2">
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Train Mean</p>
                <p className="text-sm font-mono text-slate-300">{check.details.train_mean?.toFixed(6)}</p>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Test Mean</p>
                <p className="text-sm font-mono text-slate-300">{check.details.test_mean?.toFixed(6)}</p>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Train Std</p>
                <p className="text-sm font-mono text-slate-300">{check.details.train_std?.toFixed(6)}</p>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Test Std</p>
                <p className="text-sm font-mono text-slate-300">{check.details.test_std?.toFixed(6)}</p>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Mean Diff</p>
                <p className="text-sm font-mono text-slate-300">{check.details.mean_diff?.toFixed(6)}</p>
              </div>
              <div className="bg-slate-900/50 rounded p-2">
                <p className="text-xs text-slate-500">Std Ratio</p>
                <p className="text-sm font-mono text-slate-300">{check.details.std_ratio?.toFixed(4)}</p>
              </div>
            </div>
          )}

          {/* Top Correlations */}
          {check.check_id === 'TARGET_CORRELATION' && check.details.top_correlations?.length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-slate-500 mb-1">Top feature-target correlations:</p>
              <div className="space-y-1">
                {check.details.top_correlations.map((c, i) => {
                  const absCorr = Math.abs(c.correlation)
                  const barColor = absCorr > 0.95 ? 'bg-red-500/60' : absCorr > 0.5 ? 'bg-amber-500/60' : 'bg-blue-500/60'
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="w-20 text-right text-slate-500 font-mono">feat[{c.feature_idx}]</span>
                      <div className="flex-1 h-3 bg-slate-800 rounded-sm overflow-hidden">
                        <div
                          className={`h-full ${barColor} rounded-sm`}
                          style={{ width: `${absCorr * 100}%` }}
                        />
                      </div>
                      <span className="w-16 text-right font-mono text-slate-400">
                        {c.correlation > 0 ? '+' : ''}{c.correlation.toFixed(4)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* Exact Overlap details */}
          {check.check_id === 'EXACT_OVERLAP' && (
            <div className="mt-2 text-xs text-slate-400 space-y-1">
              <p>Train size: <span className="font-mono text-slate-300">{check.details.train_size?.toLocaleString()}</span></p>
              <p>Test size: <span className="font-mono text-slate-300">{check.details.test_size?.toLocaleString()}</span></p>
              <p>Overlapping: <span className={`font-mono ${check.details.num_overlapping > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                {check.details.num_overlapping}
              </span></p>
            </div>
          )}

          {/* Near Duplicates */}
          {check.check_id === 'NEAR_DUPLICATES' && (
            <div className="mt-2 text-xs text-slate-400 space-y-1">
              <p>Threshold: <span className="font-mono text-slate-300">{check.details.threshold}</span></p>
              <p>Sampled: <span className="font-mono text-slate-300">{check.details.train_sampled?.toLocaleString()} train, {check.details.test_sampled?.toLocaleString()} test</span></p>
              <p>Near duplicates found: <span className={`font-mono ${check.details.num_near_duplicates > 0 ? 'text-orange-400' : 'text-emerald-400'}`}>
                {check.details.num_near_duplicates}
              </span></p>
              {check.details.pairs?.length > 0 && (
                <div className="mt-1">
                  <p className="text-slate-500 mb-1">Sample pairs:</p>
                  {check.details.pairs.slice(0, 5).map((p, i) => (
                    <p key={i} className="font-mono pl-2">
                      train[{p.train_idx}] ↔ test[{p.test_idx}] sim={p.similarity?.toFixed(6)}
                    </p>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Duplicates within set */}
          {(check.check_id === 'TRAIN_DUPLICATES' || check.check_id === 'TEST_DUPLICATES') && (
            <div className="mt-2 text-xs text-slate-400 space-y-1">
              {check.details.unique_samples != null && (
                <p>Unique samples: <span className="font-mono text-slate-300">{check.details.unique_samples?.toLocaleString()}</span></p>
              )}
              <p>Duplicate groups: <span className={`font-mono ${check.details.num_duplicate_groups > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {check.details.num_duplicate_groups}
              </span></p>
              <p>Extra copies: <span className="font-mono text-slate-300">{check.details.total_extra_copies}</span></p>
            </div>
          )}

          {/* Raw details fallback */}
          {!['LABEL_DISTRIBUTION', 'FEATURE_STATS', 'TARGET_CORRELATION', 'EXACT_OVERLAP', 'NEAR_DUPLICATES', 'TRAIN_DUPLICATES', 'TEST_DUPLICATES'].includes(check.check_id) && (
            <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-2 overflow-x-auto max-h-48 overflow-y-auto mt-2">
              {JSON.stringify(check.details, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  )
}

export default function LeakageReport() {
  const { runId } = useParams()
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedChecks, setExpandedChecks] = useState(new Set())

  useEffect(() => {
    fetchLeakageReport(runId)
      .then(data => {
        setReport(data)
        setLoading(false)
        // Auto-expand failed checks
        const failed = new Set()
        for (const r of (data.results || [])) {
          if (!r.passed) failed.add(r.check_id)
        }
        setExpandedChecks(failed)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  const toggleCheck = (checkId) => {
    setExpandedChecks(prev => {
      const next = new Set(prev)
      if (next.has(checkId)) next.delete(checkId)
      else next.add(checkId)
      return next
    })
  }

  if (loading) return <LoadingSpinner />

  if (error) {
    return (
      <div className="card text-center py-16">
        <Search className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <h3 className="text-lg font-medium text-slate-400">No Leakage Report</h3>
        <p className="text-sm text-slate-500 mt-2 max-w-md mx-auto">
          No data leakage analysis has been generated for this run yet.
          Add leakage detection to your training script:
        </p>
        <pre className="text-xs text-slate-400 bg-slate-900/50 rounded p-3 mt-4 max-w-lg mx-auto text-left">
{`# From DataLoaders:
run.check_leakage_from_loaders(train_loader, test_loader)

# From numpy arrays:
run.check_leakage(train_x, train_y, test_x, test_y)`}
        </pre>
      </div>
    )
  }

  if (!report) return <ErrorMessage message="No leakage report data" />

  const results = report.results || []
  const numPassed = report.num_passed || 0
  const numFailed = report.num_failed || 0

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Shield className="w-6 h-6 text-glass-500" />
            Data Leakage Report
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            {results.length} checks • {report.total_duration_ms?.toFixed(0)}ms total
          </p>
        </div>
        <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
          report.passed
            ? 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/30'
            : 'bg-red-400/10 text-red-400 border border-red-400/30'
        }`}>
          {report.passed ? '✓ All Checks Passed' : `✗ ${numFailed} Check${numFailed !== 1 ? 's' : ''} Failed`}
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <div className="card text-center">
          <div className="text-2xl font-bold">{results.length}</div>
          <div className="text-xs text-slate-500 uppercase tracking-wider">Total Checks</div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-emerald-400">{numPassed}</div>
          <div className="text-xs text-emerald-400/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <CheckCircle className="w-3 h-3" /> Passed
          </div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-red-400">{numFailed}</div>
          <div className="text-xs text-red-400/70 uppercase tracking-wider flex items-center justify-center gap-1">
            <XCircle className="w-3 h-3" /> Failed
          </div>
        </div>
        <div className="card text-center">
          <div className="text-2xl font-bold text-slate-300">{report.total_duration_ms?.toFixed(0)}</div>
          <div className="text-xs text-slate-500 uppercase tracking-wider">Duration (ms)</div>
        </div>
      </div>

      {/* Check results */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium text-slate-400 flex items-center gap-2">
            <Shield className="w-4 h-4 text-glass-500" />
            Leakage Checks
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setExpandedChecks(new Set(results.map(r => r.check_id)))}
              className="text-xs px-3 py-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Expand All
            </button>
            <button
              onClick={() => setExpandedChecks(new Set())}
              className="text-xs px-3 py-1 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Collapse All
            </button>
          </div>
        </div>

        <div className="space-y-2">
          {results.map(check => (
            <CheckCard
              key={check.check_id}
              check={check}
              isExpanded={expandedChecks.has(check.check_id)}
              onToggle={() => toggleCheck(check.check_id)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
