import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { fetchCheckpoints } from '../api'
import { LoadingSpinner, ErrorMessage } from '../components/ui'
import { GitCompare, Tag, HardDrive } from 'lucide-react'

export default function CheckpointBrowser() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const [checkpoints, setCheckpoints] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState([])

  useEffect(() => {
    fetchCheckpoints(runId)
      .then(data => {
        setCheckpoints(data.checkpoints || [])
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  const toggleSelect = (step) => {
    setSelected(prev => {
      if (prev.includes(step)) return prev.filter(s => s !== step)
      if (prev.length >= 2) return [prev[1], step]
      return [...prev, step]
    })
  }

  const compareDiff = () => {
    if (selected.length === 2) {
      const [a, b] = selected.sort((x, y) => x - y)
      navigate(`/run/${encodeURIComponent(runId)}/diff?a=${a}&b=${b}`)
    }
  }

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Checkpoints</h1>
          <p className="text-sm text-slate-500 mt-1">{checkpoints.length} checkpoint{checkpoints.length !== 1 ? 's' : ''} captured</p>
        </div>

        {selected.length === 2 && (
          <button
            onClick={compareDiff}
            className="flex items-center gap-2 px-4 py-2 rounded-lg bg-glass-600 text-white text-sm hover:bg-glass-700 transition-colors"
          >
            <GitCompare className="w-4 h-4" />
            Compare Selected
          </button>
        )}
      </div>

      {selected.length > 0 && selected.length < 2 && (
        <div className="card mb-4 border-glass-500/30 bg-glass-600/5">
          <p className="text-sm text-glass-300">
            Select one more checkpoint to compare. Selected: step {selected[0]?.toLocaleString()}
          </p>
        </div>
      )}

      <div className="space-y-2">
        {checkpoints.map((ckpt, i) => {
          const isSelected = selected.includes(ckpt.step)
          return (
            <div
              key={i}
              onClick={() => toggleSelect(ckpt.step)}
              className={`card cursor-pointer transition-all ${
                isSelected
                  ? 'border-glass-500/50 bg-glass-600/10'
                  : 'hover:border-slate-700 hover:bg-slate-900/80'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className={`w-4 h-4 rounded border-2 flex items-center justify-center ${
                    isSelected ? 'border-glass-500 bg-glass-500' : 'border-slate-600'
                  }`}>
                    {isSelected && <span className="text-white text-xs">✓</span>}
                  </div>

                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono font-medium text-white">Step {ckpt.step?.toLocaleString()}</span>
                      {ckpt.tag && (
                        <span className="flex items-center gap-1 text-xs text-glass-400 bg-glass-600/10 px-2 py-0.5 rounded-md">
                          <Tag className="w-3 h-3" />
                          {ckpt.tag}
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-slate-500">{ckpt.timestamp_str || ''}</span>
                  </div>
                </div>

                <div className="flex items-center gap-6 text-sm text-slate-400">
                  <span className="flex items-center gap-1">
                    <HardDrive className="w-3.5 h-3.5" />
                    {ckpt.size_mb || '?'} MB
                  </span>
                  <span className="font-mono text-xs text-slate-500">
                    {ckpt.num_params?.toLocaleString() || '?'} params
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
