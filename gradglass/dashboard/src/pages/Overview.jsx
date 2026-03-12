import React, { useEffect, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import { useTheme } from '../components/ThemeProvider';
import ReactECharts from 'echarts-for-react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Zap, Target, BookOpen, Clock } from 'lucide-react';

export default function Overview() {
  const { runId } = useParams();
  const { setActiveRun, metadata, metrics, alerts } = useRunStore();
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) {
      setActiveRun(runId);
    }
  }, [runId, setActiveRun]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';

  const steps = metrics.map(m => m.step);
  const currentStep = steps.length > 0 ? Math.max(...steps) : 0;
  
  const hasAlerts = alerts && alerts.length > 0;
  const statusColor = hasAlerts ? 'text-amber-600 dark:text-amber-500' : 'text-emerald-600 dark:text-emerald-500';
  const StatusIcon = hasAlerts ? ShieldAlert : ShieldCheck;

  // Generic data extractor
  const extractSeries = (key) => {
    const data = metrics.filter(m => key in m && m[key] !== null).map(m => [m.step, m[key]]);
    return data.length > 0 ? data : null;
  };

  const trainLoss = extractSeries('loss') || [];
  const valLoss = extractSeries('val_loss') || [];
  const lrData = extractSeries('lr') || [];
  
  // Advanced metrics (LLM / RL / Performance)
  const perplexity = extractSeries('perplexity') || [];
  const throughput = extractSeries('tokens_per_sec') || extractSeries('throughput') || [];
  const reward = extractSeries('reward') || extractSeries('mean_reward') || [];
  const klDiv = extractSeries('kl_divergence') || extractSeries('kl') || [];

  const commonChartOptions = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: 50, right: 30, bottom: 30, top: 30, containLabel: false },
    xAxis: { type: 'value', name: 'Step', nameLocation: 'middle', nameGap: 25, splitLine: { show: false }, axisLabel: { color: textColor } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } }
  };

  const lossOptions = {
    ...commonChartOptions,
    legend: { data: ['Train Loss', 'Val Loss'], textStyle: { color: textColor }, top: 0, right: 0 },
    series: [
      { name: 'Train Loss', type: 'line', data: trainLoss, showSymbol: false, smooth: true, itemStyle: { color: '#FDA481' } }, // Accuracy/Positive: Accent Orange
      ...(valLoss.length > 0 ? [{ name: 'Val Loss', type: 'line', data: valLoss, showSymbol: false, smooth: true, itemStyle: { color: '#B4182D' } }] : []) // Warnings/Drift: Primary Red
    ],
  };

  const performanceOptions = {
    ...commonChartOptions,
    yAxis: { ...commonChartOptions.yAxis, name: 'Tokens/sec' },
    series: [
      { name: 'Throughput', type: 'line', data: throughput, showSymbol: false, smooth: true, itemStyle: { color: '#37415C' }, areaStyle: { opacity: 0.1, color: '#37415C' } }
    ]
  };

  const lrOptions = {
    ...commonChartOptions,
    yAxis: { type: 'log', name: 'Learning Rate', splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
    series: [
      { name: 'LR', type: 'line', data: lrData, showSymbol: false, step: 'end', itemStyle: { color: '#FDA481' } }
    ]
  };

  const advancedOptions = {
    ...commonChartOptions,
    legend: { textStyle: { color: textColor }, top: 0, right: 0 },
    series: [
      ...(perplexity.length ? [{ name: 'Perplexity', type: 'line', data: perplexity, showSymbol: false, smooth: true, itemStyle: { color: '#B4182D' } }] : []),
      ...(reward.length ? [{ name: 'Reward', type: 'line', data: reward, showSymbol: false, smooth: true, itemStyle: { color: '#FDA481' } }] : []),
      ...(klDiv.length ? [{ name: 'KL Div', type: 'line', data: klDiv, showSymbol: false, smooth: true, itemStyle: { color: '#37415C' } }] : []),
    ]
  };

  if (!metadata) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading run data...</div>;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="h2 text-theme-text-primary">Global Experiment Overview</h1>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border bg-white dark:bg-slate-900 shadow-sm ${hasAlerts ? 'border-amber-300 dark:border-amber-500/30' : 'border-emerald-300 dark:border-emerald-500/30'}`}>
          <StatusIcon className={`w-5 h-5 ${statusColor}`} />
          <span className={`text-sm font-bold tracking-wide ${statusColor}`}>
            SYSTEM: {hasAlerts ? 'WARNINGS DETECTED' : 'HEALTHY'}
          </span>
        </div>
      </div>

      {hasAlerts && (
         <div className="space-y-3">
           {alerts.map((a, i) => (
             <div key={i} className={`p-4 rounded-lg flex items-start gap-3 border ${a.severity === 'high' ? 'bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/20' : 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/20'}`}>
               <AlertTriangle className={`w-5 h-5 shrink-0 mt-0.5 ${a.severity === 'high' ? 'text-red-600 dark:text-red-500' : 'text-amber-600 dark:text-amber-500'}`} />
               <div>
                  <h4 className={`font-semibold text-sm ${a.severity === 'high' ? 'text-red-700 dark:text-red-400' : 'text-amber-700 dark:text-amber-500'}`}>{a.title}</h4>
                  <p className={`text-sm mt-1 ${a.severity === 'high' ? 'text-red-600 dark:text-red-400/80' : 'text-amber-600 dark:text-amber-500/80'}`}>{a.message}</p>
               </div>
             </div>
           ))}
         </div>
      )}

      {/* KPI Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Current Step</p>
            <Zap className="w-4 h-4 text-blue-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{currentStep.toLocaleString()}</p>
        </div>
        
        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
             <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Time Elapsed</p>
             <Clock className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">
            {metadata.start_time_epoch ? Math.round((Date.now()/1000) - metadata.start_time_epoch) + 's' : '-'}
          </p>
        </div>

        {throughput.length > 0 && (
          <div className="card flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2 opacity-80">
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Throughput</p>
              <Zap className="w-4 h-4 text-violet-500" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">
              {Math.round(throughput[throughput.length - 1][1]).toLocaleString()} <span className="text-lg text-slate-400 font-normal">t/s</span>
            </p>
          </div>
        )}

        {perplexity.length > 0 && (
          <div className="card flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2 opacity-80">
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Perplexity</p>
              <BookOpen className="w-4 h-4 text-rose-500" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">
              {perplexity[perplexity.length - 1][1].toFixed(2)}
            </p>
          </div>
        )}
        
        {reward.length > 0 && (
          <div className="card flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2 opacity-80">
              <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Reward</p>
              <Target className="w-4 h-4 text-amber-500" />
            </div>
            <p className="text-3xl font-bold text-slate-900 dark:text-white">
              {reward[reward.length - 1][1].toFixed(3)}
            </p>
          </div>
        )}
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card h-[380px] flex flex-col">
          <h3 className="h3 text-theme-text-primary mb-2">Training Performance (Loss)</h3>
          <div className="flex-1 min-h-0">
            <ReactECharts option={lossOptions} style={{ height: '100%', width: '100%' }} />
          </div>
        </div>

        <div className="card h-[380px] flex flex-col">
          <h3 className="h3 text-theme-text-primary mb-2">Learning Rate Schedule</h3>
          <div className="flex-1 min-h-0">
            <ReactECharts option={lrOptions} style={{ height: '100%', width: '100%' }} />
          </div>
        </div>

        {(advancedOptions.series.length > 0 || throughput.length > 0) && (
          <>
            <div className="card h-[380px] flex flex-col">
              <h3 className="h3 text-theme-text-primary mb-2">Advanced Metrics (LLM/RL)</h3>
              <div className="flex-1 min-h-0 relative">
                {advancedOptions.series.length > 0 ? (
                  <ReactECharts option={advancedOptions} style={{ height: '100%', width: '100%' }} />
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">No advanced metrics logged.</div>
                )}
              </div>
            </div>

            <div className="card h-[380px] flex flex-col">
              <h3 className="h3 text-theme-text-primary mb-2">System Throughput</h3>
              <div className="flex-1 min-h-0 relative">
                 {throughput.length > 0 ? (
                   <ReactECharts option={performanceOptions} style={{ height: '100%', width: '100%' }} />
                 ) : (
                   <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">No throughput/tokens_per_sec logged.</div>
                 )}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
