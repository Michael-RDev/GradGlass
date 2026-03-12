import { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { fetchEvalLab } from '../../api'
import { LoadingSpinner, ErrorMessage } from '../../components/ui'

function ConfusionMatrix({ data }) {
  if (!data || !data.matrix || !data.classes) return <div className="text-slate-500 text-sm">No confusion matrix data</div>
  
  const { classes, matrix } = data
  // finding max val to scale colors
  let maxVal = 0
  for (let r of matrix) {
    for (let c of r) {
      if (c > maxVal) maxVal = c
    }
  }

  return (
    <div className="overflow-x-auto pb-4">
      <table className="border-collapse text-sm">
        <thead>
          <tr>
            <th className="p-2 text-right text-slate-500 font-medium">True \ Pred</th>
            {classes.map(c => (
              <th key={c} className="p-2 w-12 text-center text-slate-300 font-medium border-b border-slate-700/50">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={`row-${i}`}>
              <th className="p-2 pr-4 text-right text-slate-300 border-r border-slate-700/50">
                {classes[i]}
              </th>
              {row.map((val, j) => {
                const isDiag = i === j
                const intensity = maxVal > 0 ? (val / maxVal) : 0
                // Use inline style for opacity so Tailwind JIT doesn't need to generate
                // dynamic class names like bg-emerald-500/37 at build time.
                const cellStyle = val > 0
                  ? { backgroundColor: isDiag
                      ? `rgba(16, 185, 129, ${Math.max(0.1, intensity * 0.85 + 0.1)})`   // emerald-500
                      : `rgba(239, 68, 68,  ${Math.max(0.1, intensity * 0.85 + 0.1)})` } // red-500
                  : {}

                return (
                  <td
                    key={`cell-${i}-${j}`}
                    className={`p-2 w-12 h-12 text-center align-middle font-mono border border-slate-700/30 transition-colors hover:bg-slate-700 cursor-pointer ${val === 0 ? 'bg-slate-800/20' : ''} ${val > 0 ? 'text-white font-medium' : 'text-slate-600'}`}
                    style={cellStyle}
                    title={`True: ${classes[i]}, Predicted: ${classes[j]}\nCount: ${val}`}
                  >
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function EvalLab() {
  const { runId } = useParams()
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [selectedStep, setSelectedStep] = useState(null)

  useEffect(() => {
    fetchEvalLab(runId)
      .then(res => {
        setData(res.evaluations || [])
        if (res.evaluations?.length > 0) {
          setSelectedStep(res.evaluations[res.evaluations.length - 1].step)
        }
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
     <div className="max-w-5xl mx-auto space-y-8 animate-in fade-in duration-500">
        <h1 className="text-3xl font-bold text-white mb-2">Evaluation Lab</h1>
        <div className="card text-center py-16">
          <p className="text-slate-400">No evaluation data found.</p>
          <p className="text-sm text-slate-500 mt-2">Log predictions with `run.log_batch(x, y, y_pred)`</p>
        </div>
     </div>
  )

  const currentEval = data.find(d => d.step === selectedStep) || data[data.length - 1]

  return (
    <div className="max-w-6xl mx-auto space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between mb-2">
         <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
           Evaluation Lab
         </h1>
         {data.length > 1 && (
           <div className="flex items-center gap-3">
             <span className="text-sm text-slate-500">Step:</span>
             <select 
                className="bg-slate-800 border-slate-700 text-slate-300 rounded-md text-sm px-3 py-1.5 focus:ring-indigo-500 focus:border-indigo-500 outline-none"
                value={selectedStep}
                onChange={e => setSelectedStep(Number(e.target.value))}
              >
                {data.map(d => (
                  <option key={d.step} value={d.step}>Step {d.step}</option>
                ))}
             </select>
           </div>
         )}
      </div>

      {!currentEval.is_classification ? (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
           <div className="card">
             <h3 className="text-slate-400 text-sm mb-2 uppercase tracking-wide">Root Mean Square Error</h3>
             <div className="text-3xl font-mono text-orange-400">{currentEval.rmse.toFixed(4)}</div>
           </div>
           <div className="card">
             <h3 className="text-slate-400 text-sm mb-2 uppercase tracking-wide">Mean Absolute Error</h3>
             <div className="text-3xl font-mono text-indigo-400">{currentEval.mae.toFixed(4)}</div>
           </div>
           <div className="card">
             <h3 className="text-slate-400 text-sm mb-2 uppercase tracking-wide">Mean Squared Error</h3>
             <div className="text-3xl font-mono text-slate-300">{currentEval.mse.toFixed(4)}</div>
           </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left/Top: Summary Metrics */}
          <div className="lg:col-span-4 space-y-6">
            <div className="card bg-slate-900/60 border-indigo-500/20">
               <h3 className="text-indigo-400 text-sm mb-2 uppercase tracking-wide font-semibold">Macro F1 Score</h3>
               <div className="text-4xl font-bold font-mono text-white mb-4">
                 {(currentEval.macro_f1 * 100).toFixed(1)}<span className="text-lg text-slate-500 ml-1">%</span>
               </div>
               
               <div className="space-y-3 pt-4 border-t border-slate-800">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">Accuracy</span>
                    <span className="font-mono text-emerald-400">{(currentEval.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">Macro Precision</span>
                    <span className="font-mono text-slate-300">{(currentEval.macro_precision * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">Macro Recall</span>
                    <span className="font-mono text-slate-300">{(currentEval.macro_recall * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-400 text-sm">Total Support</span>
                    <span className="font-mono text-slate-300">{currentEval.support}</span>
                  </div>
               </div>
            </div>
            
            <div className="card">
               <h3 className="text-slate-300 font-semibold mb-4">Per-Class Breakdown</h3>
               <div className="space-y-3 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                  {currentEval.per_class?.map((pc) => (
                    <div key={pc.class} className="bg-slate-800/40 p-3 rounded-lg border border-slate-700/50">
                       <div className="flex justify-between items-center mb-2">
                          <span className="font-medium text-white">Class {pc.class}</span>
                          <span className="text-xs text-slate-500 font-mono">n={pc.support}</span>
                       </div>
                       <div className="grid grid-cols-3 gap-2 text-xs">
                         <div>
                           <div className="text-slate-500 mb-0.5">F1</div>
                           <div className="font-mono text-indigo-400">{(pc.f1 * 100).toFixed(1)}</div>
                         </div>
                         <div>
                           <div className="text-slate-500 mb-0.5">Pre</div>
                           <div className="font-mono text-slate-300">{(pc.precision * 100).toFixed(1)}</div>
                         </div>
                         <div>
                           <div className="text-slate-500 mb-0.5">Rec</div>
                           <div className="font-mono text-slate-300">{(pc.recall * 100).toFixed(1)}</div>
                         </div>
                       </div>
                    </div>
                  ))}
               </div>
            </div>
          </div>

          {/* Center/Right: Confusion Matrix */}
          <div className="lg:col-span-8 flex flex-col gap-6">
            <div className="card flex-1 min-h-[500px]">
              <h3 className="text-lg font-semibold text-white mb-6">Confusion Matrix</h3>
              <div className="flex justify-center items-center w-full h-full pb-10">
                 {currentEval.confusion_matrix && (
                   <ConfusionMatrix data={currentEval.confusion_matrix} />
                 )}
              </div>
            </div>
          </div>
          
        </div>
      )}
    </div>
  )
}
