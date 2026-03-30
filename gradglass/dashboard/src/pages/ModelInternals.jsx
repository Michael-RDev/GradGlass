import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { fetchGradients, fetchFreezeCode } from '../api';
import ReactECharts from 'echarts-for-react';
import { Snowflake, Activity, Zap, Layers, Network, Share2, ZoomIn, Eye, BarChart2, Image as ImageIcon, Box } from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';
import ArchitectureGraph from './ArchitectureGraph';

export default function ModelInternals() {
  const { runId } = useParams();
  const [gradients, setGradients] = useState(null);
  const [freezeCode, setFreezeCode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('architecture');
  const [selectedDagNode, setSelectedDagNode] = useState(null);
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
    
    const layers = Object.keys(latest.layers).sort();
    
    // Reverse the layer list order to match standard visualization (input at bottom/top)
    // Actually just keep standard order but reverse the array for Echarts so top of network is top of chart
    const yAxisData = [...layers].reverse();
    const norms = yAxisData.map(l => {
      const val = latest.layers[l].norm;
      return val && !isNaN(val) ? val : 0;
    });

    const opt = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 160, right: 30, bottom: 30, top: 30 },
      xAxis: { type: 'log', name: 'Gradient Norm (Log)', splitLine: { lineStyle: { color: gridColor } }, axisLabel: { color: textColor } },
      yAxis: { type: 'category', data: yAxisData, axisLabel: { color: textColor, fontSize: 10, width: 140, overflow: 'truncate' } },
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

  const TABS = [
    { id: 'architecture', label: 'Architecture & Gradients', icon: Network },
    { id: 'ablation', label: 'Layer Ablation', icon: Snowflake },
    { id: 'distributions', label: 'Distributions', icon: BarChart2 },
    { id: 'saliency', label: 'Saliency Maps', icon: ImageIcon },
    { id: 'embeddings', label: 'Embeddings', icon: Box },
  ];

  if (loading) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading architecture data...</div>;

  return (
    <div className="space-y-6 flex flex-col h-full min-h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="h2 text-theme-text-primary">Visualizations Hub</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Deep dive into model architecture, gradients, and features.</p>
        </div>
        <div className="flex gap-4">
           <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium text-slate-700 dark:text-slate-300 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800">
             <Share2 className="w-4 h-4" /> Export Report
           </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
        <div className="card flex flex-col justify-between hover:border-theme-primary transition-colors cursor-default">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Tracked Modules</p>
             <Layers className="w-4 h-4 text-theme-primary" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{numLayers.toLocaleString() || '--'}</p>
        </div>
        
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest Step</p>
             <Zap className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{latestStep?.toLocaleString() || '--'}</p>
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

      {/* Tabs */}
      <div className="flex gap-2 border-b border-theme-border pb-px shrink-0 overflow-x-auto custom-scrollbar">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === tab.id 
                ? 'border-theme-primary text-theme-primary' 
                : 'border-transparent text-theme-text-secondary hover:text-theme-text-primary hover:border-theme-border'
            }`}
          >
            <tab.icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-theme-primary' : 'opacity-70'}`} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 min-h-0 flex flex-col gap-6">
        
        {/* ARCHITECTURE & GRADIENTS TAB */}
        {activeTab === 'architecture' && (
          <>
            <div className="card p-0 overflow-hidden flex flex-col relative h-[60vh] min-h-[500px] border-theme-border shadow-md">
              <ArchitectureGraph 
                hideHeader 
                onNodeSelect={(node) => setSelectedDagNode(node)} 
                heightClass="h-full" 
              />
              {selectedDagNode && (
                <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur border border-indigo-500/30 p-3 rounded-lg shadow-xl z-10 pointer-events-none fade-in">
                   <p className="text-xs text-indigo-300 font-mono mb-1">Selected Node</p>
                   <p className="text-sm text-white font-bold">{selectedDagNode.id}</p>
                   {selectedDagNode.param_count > 0 && (
                     <p className="text-xs text-slate-400 mt-1">{selectedDagNode.param_count.toLocaleString()} params</p>
                   )}
                </div>
              )}
            </div>

            {(!gradients?.summaries?.length) ? (
              <div className="card p-8 text-slate-500 dark:text-slate-400">No gradient data found. Run with <code className="text-pink-400">gradients='summary'</code>.</div>
            ) : (
              <div className="card shadow-md border-theme-border">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="h3 text-theme-text-primary">Global Gradient Stability</h3>
                  <span className="text-xs px-2 py-0.5 rounded-md bg-indigo-100 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">Step {latestStep}</span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 mb-6">Real-time L2 norm extraction spanning the network depth. Identify shattering, exploding (red), or vanishing (blue) phenomena immediately.</p>
                <div className="h-[400px] overflow-y-auto pr-2 custom-scrollbar border border-slate-100 dark:border-slate-800/50 rounded-lg">
                  {flowOptions && <ReactECharts option={flowOptions} style={{ height: `${Math.max(400, numLayers * 22)}px`, width: '100%' }} />}
                </div>
              </div>
            )}
          </>
        )}

        {/* LAYER ABLATION TAB */}
        {activeTab === 'ablation' && (
           <div className="card shadow-md border-theme-border">
              <h3 className="h3 text-theme-text-primary mb-4 flex items-center gap-2">
                <Snowflake className="w-5 h-5 text-blue-500" />
                Layer Ablation / Freeze Tools
              </h3>
              <p className="text-sm text-slate-600 dark:text-slate-400 mb-6 max-w-2xl">
                {freezeCode?.message || "Analyzing layers for zero to extremely low gradient activity to suggest layers you can freeze to save memory and compute."}
              </p>

              {freezeCode?.candidates && freezeCode.candidates.length > 0 ? (
                <div className="grid gap-3 grid-cols-1 md:grid-cols-2 lg:grid-cols-3 mb-6">
                  {freezeCode.candidates.map((c, i) => (
                    <div key={i} className="p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md transition-shadow">
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
                <div className="p-8 border border-slate-200 dark:border-slate-800 border-dashed rounded-xl bg-slate-50 dark:bg-slate-900/30 flex flex-col items-center justify-center text-slate-500 mb-6">
                  <Snowflake className="w-8 h-8 text-slate-400 mb-3 opacity-50" />
                  <p className="text-sm text-center">No layers are inactive enough to freeze safely across recent steps.</p>
                </div>
              )}

              {freezeCode?.pytorch_code && (
                <div className="mt-4">
                  <p className="text-xs font-semibold text-slate-500 dark:text-slate-500 uppercase tracking-wider mb-2">Automated PyTorch Patch</p>
                  <div className="bg-[#0f111a] p-5 rounded-xl overflow-x-auto text-sm font-mono text-slate-300 custom-scrollbar border border-slate-800/80 shadow-inner">
                    <pre className="leading-relaxed">{freezeCode.pytorch_code}</pre>
                  </div>
                </div>
              )}
           </div>
        )}

        {/* DISTRIBUTIONS TAB (STUB) */}
        {activeTab === 'distributions' && (
          <div className="card shadow-md border-theme-border flex flex-col items-center justify-center py-20 bg-gradient-to-br from-slate-900 to-slate-800 relative overflow-hidden">
             <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-emerald-500 to-teal-400"></div>
             <BarChart2 className="w-16 h-16 text-emerald-400/50 mb-4" />
             <h3 className="text-xl font-bold text-white mb-2">Weight & Activation Distributions</h3>
             <p className="text-slate-400 max-w-md text-center text-sm">
               Histograms and violin plots of internal layer values to identify dying ReLUs, 
               exploding weights, and outlier features.
             </p>
             <button className="mt-6 px-6 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-emerald-500/20">
               Coming Soon
             </button>
             
             {/* Decorative placeholder chart */}
             <div className="absolute bottom-0 inset-x-0 h-32 opacity-10 flex items-end justify-center gap-1 pointer-events-none">
               {[...Array(40)].map((_, i) => (
                 <div key={i} className="w-4 bg-emerald-400 rounded-t-sm" style={{ height: `${Math.random() * 100}%` }}></div>
               ))}
             </div>
          </div>
        )}

        {/* SALIENCY MAPS TAB (STUB) */}
        {activeTab === 'saliency' && (
          <div className="card shadow-md border-theme-border flex flex-col items-center justify-center py-20 bg-gradient-to-br from-slate-900 to-slate-800 relative overflow-hidden">
             <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-fuchsia-500 to-pink-500"></div>
             <ImageIcon className="w-16 h-16 text-fuchsia-400/50 mb-4" />
             <h3 className="text-xl font-bold text-white mb-2">Attention & Saliency Maps</h3>
             <p className="text-slate-400 max-w-md text-center text-sm">
               Visualize input tokens and pixels with highest attributions. Map feature gradients
               directly back to the inputs to explain model predictions.
             </p>
             <button className="mt-6 px-6 py-2 bg-fuchsia-600 hover:bg-fuchsia-500 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-fuchsia-600/20">
               Requires Gradient Hook
             </button>

             {/* Decorative placeholder */}
             <div className="absolute bottom-0 right-0 w-64 h-64 bg-fuchsia-500 rounded-full blur-[100px] opacity-10 pointer-events-none"></div>
          </div>
        )}

        {/* EMBEDDINGS TAB (STUB) */}
        {activeTab === 'embeddings' && (
          <div className="card shadow-md border-theme-border flex flex-col items-center justify-center py-20 bg-gradient-to-br from-slate-900 to-slate-800 relative overflow-hidden">
             <div className="absolute top-0 inset-x-0 h-1 bg-gradient-to-r from-sky-500 to-indigo-500"></div>
             <Box className="w-16 h-16 text-sky-400/50 mb-4" />
             <h3 className="text-xl font-bold text-white mb-2">Embedding Projections</h3>
             <p className="text-slate-400 max-w-md text-center text-sm">
               Interactive 3D UMAP and PCA scatter plots of latent representations across varying classes and inputs.
             </p>
             <button className="mt-6 px-6 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-sm font-medium transition-colors shadow-lg shadow-sky-600/20">
               Coming Soon
             </button>

             {/* Decorative placeholder */}
             <div className="absolute inset-0 opacity-10 pointer-events-none" style={{ backgroundImage: 'radial-gradient(circle at 50% 50%, white 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
          </div>
        )}

      </div>
    </div>
  );
}
