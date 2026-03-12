import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import { fetchLeakageReport } from '../api';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '../components/ThemeProvider';
import { ShieldAlert, ShieldCheck, Database, Search, ArrowRight, GitMerge, FileText, Image as ImageIcon, Music, Type } from 'lucide-react';

export default function Data() {
  const { runId } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) {
      setLoading(true);
      fetchLeakageReport(runId)
        .then(res => setReport(res))
        .catch(() => setReport(null))
        .finally(() => setLoading(false));
    }
  }, [runId]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';

  // Mock data for Modality Breakdown
  const modalityOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'item', backgroundColor: theme === 'dark' ? '#1e293b' : '#fff', textStyle: { color: theme === 'dark' ? '#f8fafc' : '#0f172a' } },
      legend: { bottom: '0%', left: 'center', textStyle: { color: textColor } },
      series: [
        {
          name: 'Modality',
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: { borderRadius: 10, borderColor: theme === 'dark' ? '#181A2F' : '#fff', borderWidth: 2 },
          label: { show: false, position: 'center' },
          emphasis: { label: { show: true, fontSize: 20, fontWeight: 'bold', color: textColor } },
          labelLine: { show: false },
          data: [
            { value: 1048, name: 'Text', itemStyle: { color: '#FDA481' } },
            { value: 735, name: 'Images', itemStyle: { color: '#37415C' } },
            { value: 580, name: 'Code', itemStyle: { color: '#54162B' } },
            { value: 300, name: 'Audio', itemStyle: { color: '#B4182D' } },
          ]
        }
      ]
    };
  }, [theme, textColor]);

  // Mock data for Token Distribution
  const tokenOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 40, right: 20, bottom: 30, top: 20 },
      xAxis: { type: 'category', data: ['0-512', '512-1k', '1k-2k', '2k-4k', '4k-8k', '8k+'], axisLabel: { color: textColor } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      series: [
        {
          name: 'Documents',
          type: 'bar',
          data: [120, 200, 150, 80, 70, 110],
          itemStyle: { color: '#FDA481', borderRadius: [4, 4, 0, 0] }
        }
      ]
    };
  }, [textColor, gridColor]);

  const pipelineStages = [
    { name: 'Raw Data', icon: Database, status: 'complete', count: '1.2B' },
    { name: 'Cleaning', icon: ShieldCheck, status: 'complete', count: '850M' },
    { name: 'Augmentation', icon: GitMerge, status: 'active', count: '900M' },
    { name: 'Tokenization', icon: Type, status: 'pending', count: '-' },
    { name: 'Loader', icon: ArrowRight, status: 'pending', count: '-' },
  ];

  if (loading) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading data diagnostics...</div>;

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
           <h1 className="h2 text-theme-text-primary">Dataset & Pipeline Monitor</h1>
           <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">End-to-end data observability, composition analysis, and leakage detection.</p>
        </div>
        {report && (
          <div className={`flex items-center gap-2 px-4 py-2 rounded-full border bg-white dark:bg-slate-900 shadow-sm ${report.passed ? 'border-emerald-300 dark:border-emerald-500/30 text-emerald-600 dark:text-emerald-500' : 'border-red-300 dark:border-red-500/30 text-red-600 dark:text-red-500'}`}>
            {report.passed ? <ShieldCheck className="w-5 h-5" /> : <ShieldAlert className="w-5 h-5" />}
            <span className="text-sm font-bold tracking-wide">
              {report.passed ? 'ALL CHECKS PASSED' : `${report.num_failed} CHECKS FAILED`}
            </span>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar space-y-6 pr-2">
        {/* Interactive Pipeline Monitor */}
        <div className="card">
           <h3 className="h3 text-theme-text-primary mb-6">Data Pipeline Stages</h3>
           <div className="flex items-center justify-between relative">
             <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-200 dark:bg-slate-800 -translate-y-1/2 z-0" />
             {pipelineStages.map((stage, idx) => {
               const Icon = stage.icon;
               const isActive = stage.status === 'active';
               const isComplete = stage.status === 'complete';
               const bgClass = isActive ? 'bg-theme-primary' : isComplete ? 'bg-theme-secondary' : 'bg-slate-200 dark:bg-slate-800';
               const borderClass = isActive ? 'ring-4 ring-theme-primary/30' : '';
               const iconColor = (isActive || isComplete) ? 'text-white' : 'text-slate-400';
               
               return (
                 <div key={idx} className="relative z-10 flex flex-col items-center gap-3">
                   <div className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${bgClass} ${borderClass}`}>
                     <Icon className={`w-5 h-5 ${iconColor}`} />
                   </div>
                   <div className="text-center">
                     <p className={`text-sm font-semibold ${isActive ? 'text-theme-primary' : 'text-theme-text-primary'}`}>{stage.name}</p>
                     <p className="text-xs font-mono text-slate-500">{stage.count} samples</p>
                   </div>
                 </div>
               )
             })}
           </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card h-[350px] flex flex-col">
             <h3 className="h3 text-theme-text-primary mb-2">Dataset Composition (Modality)</h3>
             <div className="flex-1 min-h-0">
                <ReactECharts option={modalityOptions} style={{ height: '100%', width: '100%' }} />
             </div>
          </div>
          
          <div className="card h-[350px] flex flex-col">
             <h3 className="h3 text-theme-text-primary mb-2">Token Length Distribution</h3>
             <div className="flex-1 min-h-0">
                <ReactECharts option={tokenOptions} style={{ height: '100%', width: '100%' }} />
             </div>
          </div>
        </div>

        {/* Leakage Diagnostics */}
        {report ? (
          <div className="space-y-4">
            <h3 className="h3 text-theme-text-primary">Leakage Detection Results</h3>
            
            {report.results?.map((res, i) => {
              const Icon = res.passed ? ShieldCheck : ShieldAlert;
              const bgClass = res.passed ? 'bg-slate-50 dark:bg-slate-900' : 'bg-red-50 dark:bg-red-500/10';
              const borderClass = res.passed ? 'border-slate-200 dark:border-slate-800' : 'border-red-300 dark:border-red-500/30';
              const iconColor = res.passed ? 'text-emerald-600 dark:text-emerald-500' : 'text-red-500 dark:text-red-500';
              const titleColor = res.passed ? 'text-slate-800 dark:text-slate-200' : 'text-red-600 dark:text-red-400';
              
              return (
                <div key={i} className={`p-5 rounded-xl border ${bgClass} ${borderClass} flex items-start gap-4 transition-colors`}>
                   <div className={`p-2 rounded-lg bg-white dark:bg-slate-800 shadow-sm border border-slate-100 dark:border-slate-700 ${iconColor} shrink-0`}>
                     <Icon className="w-6 h-6" />
                   </div>
                   <div className="flex-1 w-full min-w-0">
                     <div className="flex items-center justify-between mb-1">
                       <h4 className={`text-base font-semibold ${titleColor}`}>{res.title}</h4>
                       <span className={`text-xs px-2 py-0.5 rounded-full uppercase font-bold tracking-wider ${
                         res.severity === 'high' ? 'bg-red-500 text-white' : 
                         res.severity === 'medium' ? 'bg-amber-500 text-white' :
                         res.severity === 'low' ? 'bg-blue-500 text-white' :
                         'bg-slate-200 dark:bg-slate-700 text-slate-700 dark:text-slate-300'
                       }`}>
                         {res.severity}
                       </span>
                     </div>
                     
                     <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">{res.description}</p>
                     
                     {!res.passed && res.recommendation && (
                       <div className="mt-3 p-3 bg-red-100 dark:bg-red-500/10 rounded-lg border border-red-200 dark:border-red-500/20">
                         <p className="text-sm font-semibold text-red-600 dark:text-red-400 mb-1">Recommendation:</p>
                         <p className="text-sm text-red-600 dark:text-red-300">{res.recommendation}</p>
                       </div>
                     )}

                     {res.details && Object.keys(res.details).length > 0 && (
                       <div className="mt-3 p-3 bg-white dark:bg-slate-950 rounded-lg border border-slate-200 dark:border-slate-800 overflow-x-auto text-xs font-mono text-slate-500 shadow-inner">
                         <pre className="custom-scrollbar">
                           {JSON.stringify(res.details, null, 2)}
                         </pre>
                       </div>
                     )}
                   </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="p-8 border border-slate-200 dark:border-slate-800 border-dashed rounded-lg bg-slate-50 dark:bg-slate-900/30 flex flex-col items-center justify-center text-slate-500 mt-6">
             <p>No data leakage report found.</p>
             <p className="mt-2 text-sm text-slate-500">Run <code className="text-indigo-500">run.check_leakage(train_x, train_y, test_x, test_y)</code> to generate a report.</p>
          </div>
        )}
      </div>
    </div>
  );
}
