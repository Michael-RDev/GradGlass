import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { fetchGradients, fetchFreezeCode } from '../api';
import ReactECharts from 'echarts-for-react';
import { Snowflake, Activity, Zap, Layers, Share2, ZoomIn, Eye } from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';

export default function ModelInternals() {
  const { runId } = useParams();
  const [gradients, setGradients] = useState(null);
  const [freezeCode, setFreezeCode] = useState(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) {
      setLoading(true);
      Promise.all([
        fetchGradients(runId).catch(() => null),
        fetchFreezeCode(runId).catch(() => null)
      ])
      .then(([gradRes, freezeRes]) => {
        if (gradRes) setGradients(gradRes);
        if (freezeRes) setFreezeCode(freezeRes);
      })
      .finally(() => setLoading(false));
    }
  }, [runId]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';

  const { latestStep, numLayers, flowOptions, layersList } = useMemo(() => {
    if (!gradients || !gradients.summaries || gradients.summaries.length === 0) {
      return { latestStep: 0, numLayers: 0, flowOptions: null, layersList: [] };
    }
    const summaries = gradients.summaries;
    const latest = summaries[summaries.length - 1];
    
    // Sort layers by name roughly (top to bottom of network)
    const layers = Object.keys(latest.layers).sort();
    
    const norms = layers.map(l => {
      const val = latest.layers[l].norm;
      return val && !isNaN(val) ? val : 0;
    });

    const opt = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 150, right: 30, bottom: 30, top: 30 },
      xAxis: { type: 'log', name: 'Gradient Norm (Log)', splitLine: { lineStyle: { color: gridColor } }, axisLabel: { color: textColor } },
      yAxis: { type: 'category', data: layers, axisLabel: { color: textColor, fontSize: 10, width: 130, overflow: 'truncate' } },
      series: [
        { 
          name: 'L2 Norm', 
          type: 'bar', 
          data: norms, 
          itemStyle: { 
            color: (params) => {
              const val = params.value;
              if (val < 1e-5) return '#37415C'; // vanishing: Muted Slate
              if (val > 10) return '#B4182D'; // exploding: Primary Red
              return '#FDA481'; // healthy: Accent Orange
            }
          } 
        }
      ]
    };

    return { latestStep: latest.step, numLayers: layers.length, flowOptions: opt, layersList: layers.map((name, i) => ({ name, norm: norms[i] })) };
  }, [gradients, theme, gridColor, textColor]);

  if (loading) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading architecture data...</div>;
  if (!gradients?.summaries?.length) return <div className="p-8 text-slate-500 dark:text-slate-400">No gradient data found. Run with <code className="text-pink-400">gradients='summary'</code>.</div>;

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="h2 text-theme-text-primary">Model Architecture Explorer</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Layer-by-layer observability, FLOPs tracking, and gradient stability analysis.</p>
        </div>
        <div className="flex gap-4">
           {/* Placeholder for architecture level controls */}
           <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium text-slate-700 dark:text-slate-300 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800">
             <Share2 className="w-4 h-4" /> Export Graph
           </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Tracked Modules</p>
             <Layers className="w-4 h-4 text-indigo-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{numLayers.toLocaleString()}</p>
        </div>
        
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest Step</p>
             <Zap className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{latestStep.toLocaleString()}</p>
        </div>
        
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Est. Parameters</p>
             <ZoomIn className="w-4 h-4 text-violet-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">~{(numLayers * 1.5).toFixed(1)}M</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Ablation Candidates</p>
             <Snowflake className="w-4 h-4 text-blue-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{freezeCode?.candidates?.length || 0}</p>
        </div>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left pane: Module explorer (Tree/List) */}
        <div className="w-80 shrink-0 card flex flex-col p-0 overflow-hidden">
           <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900/50">
             <h3 className="text-sm font-semibold text-slate-900 dark:text-white">Architecture Flow</h3>
           </div>
           <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
             {layersList.map((layer, i) => (
                <div key={i} className="group flex items-center justify-between p-2 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800/50 cursor-pointer">
                  <div className="flex items-center gap-2 min-w-0">
                     <span className="text-xs font-mono text-slate-400 w-4 text-right shrink-0">{i}</span>
                     <span className="text-sm text-slate-700 dark:text-slate-300 font-mono truncate">{layer.name}</span>
                  </div>
                  <Eye className="w-4 h-4 text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                </div>
             ))}
           </div>
        </div>

        {/* Middle pane: Interactive Visualizations */}
        <div className="flex-1 overflow-y-auto custom-scrollbar space-y-6">
           <div className="card">
             <div className="flex items-center justify-between mb-2">
                <h3 className="h3 text-theme-text-primary">Global Gradient Stability</h3>
                <span className="text-xs px-2 py-0.5 rounded-md bg-indigo-100 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">Step {latestStep}</span>
             </div>
             <p className="text-xs text-slate-500 dark:text-slate-400 mb-6">Real-time L2 norm extraction spanning the network depth. Identify shattering, exploding (red), or vanishing (blue) phenomena immediately.</p>
             <div className="h-[500px] overflow-y-auto pr-2 custom-scrollbar border border-slate-100 dark:border-slate-800/50 rounded-lg">
               {flowOptions && <ReactECharts option={flowOptions} style={{ height: `${Math.max(500, numLayers * 25)}px`, width: '100%' }} />}
             </div>
           </div>

           <div className="card">
              <h3 className="h3 text-theme-text-primary mb-4 flex items-center gap-2">
                <Snowflake className="w-5 h-5 text-theme-accent" />
                Layer Ablation / Freeze Tools
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-6 max-w-2xl">
                {freezeCode?.message || "Analyzing layers for zero to extremely low gradient activity to suggest layers you can freeze to save memory and compute."}
              </p>

              {freezeCode?.candidates && freezeCode.candidates.length > 0 ? (
                <div className="grid gap-3 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 mb-6">
                  {freezeCode.candidates.map((c, i) => (
                    <div key={i} className="p-4 bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800">
                       <p className="text-sm font-mono text-slate-800 dark:text-slate-200 truncate mb-3" title={c.layer}>{c.layer}</p>
                       <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-slate-500">Relative Activity</span>
                          <span className="text-xs font-mono font-medium text-blue-600 dark:text-blue-400">{(c.relative_norm * 100).toFixed(4)}%</span>
                       </div>
                       <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div className="bg-blue-500 h-full rounded-full transition-all" style={{ width: `${Math.max(1, c.relative_norm * 100)}%` }} />
                       </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="p-8 border border-slate-200 dark:border-slate-800 border-dashed rounded-lg bg-slate-50 dark:bg-slate-900/30 flex flex-col items-center justify-center text-slate-500 mb-6">
                  <Snowflake className="w-8 h-8 text-slate-400 mb-3 opacity-50" />
                  <p className="text-sm text-center">No layers are inactive enough to freeze safely across recent steps.</p>
                </div>
              )}

              {freezeCode?.pytorch_code && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-500 uppercase tracking-wider mb-2">Automated PyTorch Patch</p>
                  <div className="bg-slate-900 p-4 rounded-lg overflow-x-auto text-sm font-mono text-slate-300 custom-scrollbar border border-slate-800 shadow-inner">
                    <pre>{freezeCode.pytorch_code}</pre>
                  </div>
                </div>
              )}
           </div>
        </div>
      </div>
    </div>
  );
}
