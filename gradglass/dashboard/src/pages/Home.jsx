import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { fetchRuns } from '../api'
import { StatusBadge, LoadingSpinner, EmptyState } from '../components/ui'
import { Database, Clock, TrendingDown, Percent } from 'lucide-react'

export default function Home() {
  const [runs, setRuns] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sortBy, setSortBy] = useState('start_time')

  useEffect(() => {
    fetchRuns()
      .then(data => {
        setRuns(data.runs || [])
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) return <LoadingSpinner />

  if (error) {
    return (
      <div className="text-center py-20">
        <h2 className="text-xl text-theme-text-secondary">Could not connect to GradGlass server</h2>
        <p className="text-sm text-theme-text-muted mt-2">{error}</p>
        <p className="text-sm text-theme-text-muted mt-4">Make sure the server is running on localhost:8432</p>
      </div>
    )
  }

  if (runs.length === 0) {
    return (
      <EmptyState
        icon={Database}
        title="No runs found"
        description="Start capturing runs with: run = gg.run('my-experiment').watch(model)"
      />
    )
  }

  const sortedRuns = [...runs].sort((a, b) => {
    if (sortBy === 'start_time') return (b.start_time_epoch || 0) - (a.start_time_epoch || 0)
    if (sortBy === 'loss') return (a.latest_loss || Infinity) - (b.latest_loss || Infinity)
    if (sortBy === 'storage') return (b.storage_bytes || 0) - (a.storage_bytes || 0)
    return 0
  })

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="h2 text-theme-text-primary">All Runs</h1>
          <p className="text-sm text-theme-text-secondary mt-1">{runs.length} experiment{runs.length !== 1 ? 's' : ''} found</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-theme-text-secondary">Sort by:</span>
          {['start_time', 'loss', 'storage'].map(key => (
            <button
              key={key}
              onClick={() => setSortBy(key)}
              className={`text-xs px-3 py-1 rounded-lg transition-colors
                ${sortBy === key ? 'bg-theme-primary/20 text-theme-primary' : 'text-theme-text-secondary hover:text-theme-text-primary hover:bg-theme-surface-hover'}`}
            >
              {key === 'start_time' ? 'Date' : key.charAt(0).toUpperCase() + key.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Run cards */}
      <div className="grid gap-3">
        {sortedRuns.map(run => (
          <Link
            key={run.run_id}
            to={`/run/${encodeURIComponent(run.run_id)}`}
            className="block card hover:border-theme-border transition-all group"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 min-w-0">
                <div className="min-w-0">
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold text-theme-text-primary group-hover:text-theme-primary transition-colors truncate">
                      {run.name}
                    </h3>
                    <StatusBadge status={run.status} />
                    {run.framework && (
                      <span className="text-xs text-theme-text-secondary bg-theme-bg border border-theme-border px-2 py-0.5 rounded-md font-mono">
                        {run.framework}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-4 mt-1.5 text-xs text-theme-text-secondary">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {run.start_time || 'Unknown'}
                    </span>
                    <span className="font-mono">{run.run_id}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-8 text-sm shrink-0">
                {run.total_steps != null && (
                  <div className="text-right">
                    <div className="text-xs text-theme-text-secondary">Steps</div>
                    <div className="font-mono font-medium text-theme-text-primary">{run.total_steps?.toLocaleString()}</div>
                  </div>
                )}
                {run.latest_loss != null && (
                  <div className="text-right">
                    <div className="text-xs text-theme-text-secondary">Loss</div>
                    <div className="font-mono font-medium flex items-center gap-1 text-theme-text-primary">
                      <TrendingDown className="w-3 h-3 text-emerald-500 dark:text-emerald-400" />
                      {run.latest_loss.toFixed(4)}
                    </div>
                  </div>
                )}
                {run.latest_acc != null && (
                  <div className="text-right">
                    <div className="text-xs text-theme-text-secondary">Acc</div>
                    <div className="font-mono font-medium flex items-center gap-1 text-theme-text-primary">
                      <Percent className="w-3 h-3 text-blue-500 dark:text-blue-400" />
                      {(run.latest_acc * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
                <div className="text-right">
                  <div className="text-xs text-theme-text-secondary">Storage</div>
                  <div className="font-mono font-medium text-theme-text-primary">
                    {run.storage_mb} MB
                  </div>
                </div>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
