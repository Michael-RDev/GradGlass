import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '../components/ThemeProvider';
import { Search, BrainCircuit, ScanSearch, Lightbulb, AlertCircle } from 'lucide-react';

export default function Interpretability() {
  const [activeTab, setActiveTab] = useState('attention');
  const { theme } = useTheme();

  const textColor = theme === 'dark' ? '#94a3b8' : '#64748b';
  const gridColor = theme === 'dark' ? '#1e293b' : '#e2e8f0';

  // Mock data for Attention Map
  const attentionOptions = useMemo(() => {
    const tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
    const data = [];
    tokens.forEach((yt, i) => {
      tokens.forEach((xt, j) => {
        // Mock diagonal-ish attention with some scatter
        let val = 0;
        if (i === j) val = 0.8 + Math.random() * 0.2;
        else if (j < i) val = (0.2 * Math.random()) / (i - j);
        else val = 0.05 * Math.random();
        data.push([j, i, val]);
      });
    });

    return {
      tooltip: { position: 'top' },
      grid: { left: 60, right: 30, bottom: 60, top: 20 },
      xAxis: { type: 'category', data: tokens, axisLabel: { interval: 0, rotate: 45, color: textColor } },
      yAxis: { type: 'category', data: tokens, axisLabel: { color: textColor } },
      visualMap: { min: 0, max: 1, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', textStyle: { color: textColor }, inRange: { color: theme === 'dark' ? ['#0f172a', '#fbbf24', '#ef4444'] : ['#f8fafc', '#fcd34d', '#dc2626'] } },
      series: [{ name: 'Attention Weight', type: 'heatmap', data, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } } }]
    };
  }, [theme, textColor]);

  // Mock data for Feature Attribution (SHAP / Integrated Gradients)
  const attributionOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 100, right: 40, bottom: 30, top: 20 },
      xAxis: { type: 'value', name: 'Mean SHAP Value', nameLocation: 'middle', nameGap: 25, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      yAxis: { type: 'category', data: ['Token "fox"', 'Token "jumps"', 'Pixel (14, 28)', 'Feature_42', 'Channel_1'], axisLabel: { color: textColor } },
      series: [
        {
          name: 'Impact on Output',
          type: 'bar',
          data: [0.85, 0.62, 0.45, -0.32, -0.55],
          itemStyle: {
            color: (params) => {
              return params.value > 0 ? '#10b981' : '#ef4444'; // Green for positive, Red for negative
            }
          }
        }
      ]
    };
  }, [textColor, gridColor]);

  const predictions = [
    { id: 1024, input: 'A bright sunny afternoon in the park...', pred: 'Positive (0.91)', actual: 'Negative', conf: '91%', loss: 2.45 },
    { id: 89, input: 'The stock market crashed entirely...', pred: 'Negative (0.85)', actual: 'Positive', conf: '85%', loss: 1.98 },
    { id: 412, input: 'I feel thoroughly ambivalent about this.', pred: 'Neutral (0.45)', actual: 'Positive', conf: '45%', loss: 1.12 },
  ];

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
           <h1 className="text-2xl font-bold text-slate-900 dark:text-white tracking-tight">Interpretability & Debugging</h1>
           <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Peek inside the black box with SHAP, attention maps, and failure analysis.</p>
        </div>
      </div>

      <div className="flex border-b border-slate-200 dark:border-slate-800 shrink-0">
         <button 
           onClick={() => setActiveTab('attention')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'attention' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <BrainCircuit className="w-4 h-4" /> Attention Maps
         </button>
         <button 
           onClick={() => setActiveTab('shap')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'shap' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <ScanSearch className="w-4 h-4" /> Feature Attribution
         </button>
         <button 
           onClick={() => setActiveTab('failures')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'failures' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <AlertCircle className="w-4 h-4" /> Worst Predictions
         </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar space-y-6 pr-2">
        {activeTab === 'attention' && (
           <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-full min-h-[500px]">
             <div className="lg:col-span-2 card flex flex-col">
                <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">Self-Attention Heatmap (Head 4, Layer 12)</h3>
                <div className="flex-1 min-h-[400px]">
                  <ReactECharts option={attentionOptions} style={{ height: '100%', width: '100%' }} />
                </div>
             </div>
             <div className="card flex flex-col space-y-4">
                <h3 className="text-base font-semibold text-slate-900 dark:text-white">Head Controls</h3>
                <div>
                   <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Select Layer</label>
                   <select className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                     <option>Layer 12 (Output)</option>
                     <option>Layer 6 (Middle)</option>
                     <option>Layer 1 (Input)</option>
                   </select>
                </div>
                <div>
                   <label className="block text-sm font-medium text-slate-600 dark:text-slate-400 mb-2">Select Attention Head</label>
                   <select className="w-full bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                     <option>Head 4</option>
                     <option>Head 1</option>
                     <option>Head 8</option>
                   </select>
                </div>
                <div className="p-4 bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/20 rounded-lg mt-auto">
                   <Lightbulb className="w-5 h-5 text-amber-500 mb-2" />
                   <p className="text-sm text-amber-700 dark:text-amber-500">Head 4 exhibits strong diagonal attention, indicating it primarily focuses on local context and immediate preceding tokens.</p>
                </div>
             </div>
           </div>
        )}

        {activeTab === 'shap' && (
           <div className="card min-h-[500px] flex flex-col">
             <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">Feature Importance (Global SHAP)</h3>
             <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">Aggregated Shapley values across the validation set. Shows which tokens or pixels influence the model's decisions the most.</p>
             <div className="flex-1 min-h-[400px]">
               <ReactECharts option={attributionOptions} style={{ height: '100%', width: '100%' }} />
             </div>
           </div>
        )}

        {activeTab === 'failures' && (
           <div className="card min-h-[500px]">
             <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-6">Highest Loss Interventions</h3>
             <div className="overflow-x-auto">
               <table className="w-full text-left text-sm text-slate-600 dark:text-slate-300">
                 <thead className="bg-slate-100 dark:bg-slate-800/50 text-slate-700 dark:text-slate-200">
                   <tr>
                      <th className="px-4 py-3 rounded-tl-lg font-medium">Sample ID</th>
                      <th className="px-4 py-3 font-medium">Input / Context</th>
                      <th className="px-4 py-3 font-medium">Prediction</th>
                      <th className="px-4 py-3 font-medium">Ground Truth</th>
                      <th className="px-4 py-3 rounded-tr-lg font-medium">Loss Score</th>
                   </tr>
                 </thead>
                 <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                   {predictions.map(p => (
                      <tr key={p.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/30">
                        <td className="px-4 py-4 font-mono text-xs">{p.id}</td>
                        <td className="px-4 py-4 max-w-xs truncate" title={p.input}>{p.input}</td>
                        <td className="px-4 py-4 text-red-600 dark:text-red-400 font-medium">{p.pred}</td>
                        <td className="px-4 py-4 text-emerald-600 dark:text-emerald-400">{p.actual}</td>
                        <td className="px-4 py-4 font-mono font-bold">{p.loss}</td>
                      </tr>
                   ))}
                 </tbody>
               </table>
             </div>
           </div>
        )}
      </div>
    </div>
  );
}
