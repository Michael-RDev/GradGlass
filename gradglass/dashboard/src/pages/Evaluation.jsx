import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { fetchEvalLab } from '../api';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '../components/ThemeProvider';
import { BarChart2, BookOpen, Image as ImageIcon, CheckCircle, Target } from 'lucide-react';

export default function Evaluation() {
  const { runId } = useParams();
  const [evalData, setEvalData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('standard');
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) {
      setLoading(true);
      fetchEvalLab(runId)
        .then(res => setEvalData(res.evaluations || []))
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [runId]);

  const textColor = theme === 'dark' ? '#94a3b8' : '#64748b';
  const gridColor = theme === 'dark' ? '#1e293b' : '#e2e8f0';

  const latestEval = evalData.length > 0 ? evalData[evalData.length - 1] : null;
  const isClass = latestEval?.is_classification;

  const trendOptions = useMemo(() => {
    if (!evalData.length) return null;
    if (isClass) {
      const steps = evalData.map(e => e.step);
      const acc = evalData.map(e => e.accuracy * 100);
      const f1 = evalData.map(e => e.macro_f1 * 100);
      return {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { data: ['Accuracy (%)', 'Macro F1 (%)'], textStyle: { color: textColor }, top: 0 },
        grid: { left: 40, right: 20, bottom: 30, top: 30 },
        xAxis: { type: 'category', data: steps, axisLabel: { color: textColor } },
        yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
        series: [
          { name: 'Accuracy (%)', type: 'line', data: acc, smooth: true, itemStyle: { color: '#3b82f6' } },
          { name: 'Macro F1 (%)', type: 'line', data: f1, smooth: true, itemStyle: { color: '#10b981' } }
        ]
      };
    } else {
      const steps = evalData.map(e => e.step);
      const mse = evalData.map(e => e.mse);
      const mae = evalData.map(e => e.mae);
      return {
        backgroundColor: 'transparent',
        tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
        legend: { data: ['MSE', 'MAE'], textStyle: { color: textColor }, top: 0 },
        grid: { left: 40, right: 20, bottom: 30, top: 30 },
        xAxis: { type: 'category', data: steps, axisLabel: { color: textColor } },
        yAxis: { type: 'value', splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
        series: [
          { name: 'MSE', type: 'line', data: mse, smooth: true, itemStyle: { color: '#3b82f6' } },
          { name: 'MAE', type: 'line', data: mae, smooth: true, itemStyle: { color: '#f43f5e' } }
        ]
      };
    }
  }, [evalData, isClass, theme, textColor, gridColor]);

  const cmOptions = useMemo(() => {
    if (!latestEval || !latestEval.confusion_matrix) return null;
    const { classes, matrix } = latestEval.confusion_matrix;
    
    const data = [];
    matrix.forEach((row, i) => {
      row.forEach((col, j) => {
        data.push([j, i, col]);
      });
    });

    const maxVal = Math.max(...data.map(d => d[2]));

    return {
      tooltip: { position: 'top' },
      grid: { left: 60, right: 20, bottom: 40, top: 10 },
      xAxis: { type: 'category', data: classes, name: 'Predicted', nameLocation: 'middle', nameGap: 25, splitArea: { show: true }, axisLabel: { color: textColor } },
      yAxis: { type: 'category', data: classes, name: 'Actual', nameLocation: 'middle', nameGap: 40, splitArea: { show: true }, axisLabel: { color: textColor } },
      visualMap: { min: 0, max: maxVal, calculable: true, orient: 'horizontal', left: 'center', bottom: '0%', textStyle: { color: textColor }, inRange: { color: theme === 'dark' ? ['#0f172a', '#6366f1'] : ['#f8fafc', '#4f46e5'] } },
      series: [{ name: 'Confusion Matrix', type: 'heatmap', data: data, label: { show: true, color: theme === 'dark' ? '#fff' : '#000' }, emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0, 0, 0, 0.5)' } } }]
    };
  }, [latestEval, theme, textColor]);

  // Mock data for LLM Benchmarks
  const llmBenchOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { textStyle: { color: textColor }, top: 0 },
      grid: { left: 40, right: 20, bottom: 30, top: 30 },
      xAxis: { type: 'category', data: ['MMLU', 'GSM8K', 'HellaSwag', 'TruthfulQA', 'HumanEval'], axisLabel: { color: textColor } },
      yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      series: [
        { name: '0-shot', type: 'bar', data: [45.2, 32.1, 71.5, 40.8, 25.0], itemStyle: { color: '#64748b' } },
        { name: '5-shot', type: 'bar', data: [51.4, 48.7, 78.2, 45.3, 33.5], itemStyle: { color: '#3b82f6' } }
      ]
    };
  }, [textColor, gridColor]);

  // Mock data for Vision Benchmarks
  const visionBenchOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { textStyle: { color: textColor }, top: 0 },
      grid: { left: 40, right: 20, bottom: 30, top: 30 },
      xAxis: { type: 'category', data: ['mAP', 'IoU', 'Top-1 Acc', 'Top-5 Acc'], axisLabel: { color: textColor } },
      yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      series: [
        { name: 'Baseline', type: 'bar', data: [38.5, 52.1, 74.2, 91.5], itemStyle: { color: '#64748b' } },
        { name: 'Current Run', type: 'bar', data: [42.1, 55.4, 78.9, 94.2], itemStyle: { color: '#10b981' } }
      ]
    };
  }, [textColor, gridColor]);

  if (loading) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading evaluation data...</div>;

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white tracking-tight">Evaluation & Benchmarks</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Multi-modal quality analysis and public benchmark suites.</p>
        </div>
      </div>

      <div className="flex border-b border-slate-200 dark:border-slate-800 shrink-0">
         <button 
           onClick={() => setActiveTab('standard')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'standard' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <BarChart2 className="w-4 h-4" /> Standard Metrics
         </button>
         <button 
           onClick={() => setActiveTab('llm')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'llm' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <BookOpen className="w-4 h-4" /> LLM Benchmarks
         </button>
         <button 
           onClick={() => setActiveTab('vision')}
           className={`px-4 py-3 text-sm font-medium flex items-center gap-2 border-b-2 transition-colors ${activeTab === 'vision' ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-300'}`}
         >
           <ImageIcon className="w-4 h-4" /> Vision Benchmarks
         </button>
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
        {activeTab === 'standard' && (
           <div className="space-y-6">
             {evalData.length === 0 ? (
               <div className="p-8 text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800">No evaluation data found. Use `run.log_batch()` in your training loop.</div>
             ) : (
               <>
                 <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                   {isClass ? (
                     <>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest Accuracy</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{(latestEval.accuracy * 100).toFixed(2)}%</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Macro F1</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{(latestEval.macro_f1 * 100).toFixed(2)}%</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Precision</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{(latestEval.macro_precision * 100).toFixed(2)}%</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Recall</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{(latestEval.macro_recall * 100).toFixed(2)}%</p>
                       </div>
                     </>
                   ) : (
                     <>
                        <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest MSE</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{latestEval.mse.toFixed(4)}</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest RMSE</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{latestEval.rmse.toFixed(4)}</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest MAE</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{latestEval.mae.toFixed(4)}</p>
                       </div>
                       <div className="card">
                         <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Evaluated Steps</p>
                         <p className="text-3xl font-bold text-slate-900 dark:text-white mt-2">{evalData.length}</p>
                       </div>
                     </>
                   )}
                 </div>

                 <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                   <div className="card h-[400px] flex flex-col">
                      <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">Performance Trend</h3>
                      <div className="flex-1 min-h-0">
                        {trendOptions && <ReactECharts option={trendOptions} style={{ height: '100%', width: '100%' }} />}
                      </div>
                   </div>

                   {isClass && cmOptions && (
                     <div className="card h-[400px] flex flex-col">
                        <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">Confusion Matrix (Latest)</h3>
                        <div className="flex-1 min-h-0">
                          <ReactECharts option={cmOptions} style={{ height: '100%', width: '100%' }} />
                        </div>
                     </div>
                   )}
                 </div>
               </>
             )}
           </div>
        )}

        {activeTab === 'llm' && (
           <div className="space-y-6">
             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 card h-[400px] flex flex-col">
                   <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2 flex items-center gap-2">
                     <BookOpen className="w-5 h-5 text-indigo-500" /> Language Model Benchmarks
                   </h3>
                   <div className="flex-1 min-h-0">
                      <ReactECharts option={llmBenchOptions} style={{ height: '100%', width: '100%' }} />
                   </div>
                </div>
                <div className="card flex flex-col">
                   <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-4">Benchmark Suites</h3>
                   <div className="space-y-4">
                      <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded-lg border border-slate-200 dark:border-slate-800">
                         <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-sm text-slate-800 dark:text-slate-200">MMLU</span>
                            <span className="text-xs px-2 py-0.5 bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 rounded">Passing</span>
                         </div>
                         <p className="text-xs text-slate-500 dark:text-slate-400">Massive Multitask Language Understanding.</p>
                      </div>
                      <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded-lg border border-slate-200 dark:border-slate-800">
                         <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-sm text-slate-800 dark:text-slate-200">GSM8K</span>
                            <span className="text-xs px-2 py-0.5 bg-rose-100 dark:bg-rose-500/20 text-rose-600 dark:text-rose-400 rounded">Underperforming</span>
                         </div>
                         <p className="text-xs text-slate-500 dark:text-slate-400">Grade School Math 8K. Model struggles with reasoning chains.</p>
                      </div>
                   </div>
                </div>
             </div>
           </div>
        )}

        {activeTab === 'vision' && (
           <div className="space-y-6">
             <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-2 card h-[400px] flex flex-col">
                   <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2 flex items-center gap-2">
                     <ImageIcon className="w-5 h-5 text-indigo-500" /> Vision Transformer Benchmarks
                   </h3>
                   <div className="flex-1 min-h-0">
                      <ReactECharts option={visionBenchOptions} style={{ height: '100%', width: '100%' }} />
                   </div>
                </div>
                <div className="card flex flex-col">
                   <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-4">Benchmark Suites</h3>
                   <div className="space-y-4">
                      <div className="p-3 bg-slate-50 dark:bg-slate-900/50 rounded-lg border border-slate-200 dark:border-slate-800">
                         <div className="flex items-center justify-between mb-1">
                            <span className="font-semibold text-sm text-slate-800 dark:text-slate-200">ImageNet Top-1</span>
                            <span className="text-xs px-2 py-0.5 bg-emerald-100 dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 rounded">SOTA</span>
                         </div>
                         <p className="text-xs text-slate-500 dark:text-slate-400">Zero-shot classification accuracy.</p>
                      </div>
                   </div>
                </div>
             </div>
           </div>
        )}
      </div>
    </div>
  );
}
