import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { fetchGradients } from '../api'
import { LoadingSpinner, ErrorMessage, EmptyState } from '../components/ui'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { AlertTriangle, Activity, Skull } from 'lucide-react'

const FLAG_COLORS = {
  VANISHING: { bg: 'bg-purple-400/10', text: 'text-purple-400', border: 'border-purple-400/30' },
  EXPLODING: { bg: 'bg-red-400/10', text: 'text-red-400', border: 'border-red-400/30' },
  DEAD: { bg: 'bg-slate-400/10', text: 'text-slate-400', border: 'border-slate-400/30' },
  NOISY: { bg: 'bg-amber-400/10', text: 'text-amber-400', border: 'border-amber-400/30' },
  DISTRIBUTION_SHIFT: { bg: 'bg-orange-400/10', text: 'text-orange-400', border: 'border-orange-400/30' },
}

export default function GradientFlow() {
  const { runId } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedLayer, setSelectedLayer] = useState(null)

  useEffect(() => {
    fetchGradients(runId)
      .then(d => {
        setData(d)
        setLoading(false)
        if (d.analysis?.length > 0) setSelectedLayer(d.analysis[0])
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [runId])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />
  if (!data?.analysis?.length) {
    return (
      <EmptyState
        icon={Activity}
        title="No gradient data"
        description="Gradient summaries are captured every N steps during training."
      />
    )
  }

  const { analysis } = data
  const flaggedLayers = analysis.filter(l => l.flags.length > 0)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Gradient Flow</h1>
          <p className="text-sm text-slate-500 mt-1">Per-layer gradient analysis across training</p>
        </div>
      </div>

      {/* Alerts */}
      {flaggedLayers.map((layer, i) => (
        <div key={i} className="card border-amber-500/20 mb-3">
          <div className="flex items-start gap-3">
            {layer.flags.includes('DEAD') ? (
              <Skull className="w-5 h-5 text-slate-400 shrink-0 mt-0.5" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            )}
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm text-white">{layer.layer}</span>
                {layer.flags.map(flag => {
                  const colors = FLAG_COLORS[flag] || FLAG_COLORS.NOISY
                  return (
                    <span key={flag} className={`badge ${colors.bg} ${colors.text} ${colors.border}`}>
                      {flag}
                    </span>
                  )
                })}
              </div>
              <p className="text-sm text-slate-400 mt-1">
                Grad mean: {layer.grad_mean.toExponential(1)} | Variance: {layer.grad_var.toExponential(1)} | KL div: {layer.kl_div.toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      ))}

      <div className="grid grid-cols-12 gap-4">
        {/* Layer table */}
        <div className="col-span-5">
          <div className="card">
            <h3 className="text-sm font-medium text-slate-400 mb-3">Per-Layer Summary</h3>
            <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-900">
                  <tr className="text-xs text-slate-500 border-b border-slate-800">
                    <th className="text-left py-2 pr-3">Layer</th>
                    <th className="text-right py-2 pr-3">Grad Mean</th>
                    <th className="text-right py-2 pr-3">Variance</th>
                    <th className="text-right py-2 pr-3">KL Div</th>
                    <th className="text-left py-2">Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.map((layer, i) => (
                    <tr
                      key={i}
                      onClick={() => setSelectedLayer(layer)}
                      className={`border-b border-slate-800/50 cursor-pointer transition-colors ${
                        selectedLayer?.layer === layer.layer
                          ? 'bg-glass-600/10'
                          : 'hover:bg-slate-800/50'
                      }`}
                    >
                      <td className="py-1.5 pr-3 font-mono text-xs truncate max-w-[150px]">{layer.layer}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-xs">{layer.grad_mean.toExponential(1)}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-xs">{layer.grad_var.toExponential(1)}</td>
                      <td className="py-1.5 pr-3 text-right font-mono text-xs">{layer.kl_div.toFixed(3)}</td>
                      <td className="py-1.5">
                        {layer.flags.map(flag => {
                          const colors = FLAG_COLORS[flag] || FLAG_COLORS.NOISY
                          return (
                            <span key={flag} className={`badge text-[10px] ${colors.bg} ${colors.text} ${colors.border}`}>
                              {flag}
                            </span>
                          )
                        })}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Gradient history chart */}
        <div className="col-span-7">
          {selectedLayer?.history?.length > 0 && (
            <div className="card">
              <h3 className="text-sm font-medium text-slate-400 mb-1">
                Gradient History — <span className="font-mono text-glass-300">{selectedLayer.layer}</span>
              </h3>
              <p className="text-xs text-slate-500 mb-4">{selectedLayer.num_steps} data points</p>

              <div className="mb-4">
                <h4 className="text-xs text-slate-500 mb-2">Gradient Norm</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={selectedLayer.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    />
                    <Line type="monotone" dataKey="norm" stroke="#6366f1" strokeWidth={2} dot={false} name="Norm" />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              <div>
                <h4 className="text-xs text-slate-500 mb-2">Gradient Mean</h4>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={selectedLayer.history}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="step" stroke="#475569" tick={{ fontSize: 10 }} />
                    <YAxis stroke="#475569" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '8px' }}
                    />
                    <Line type="monotone" dataKey="mean" stroke="#10b981" strokeWidth={2} dot={false} name="Mean" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
