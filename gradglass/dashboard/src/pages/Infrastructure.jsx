import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { useTheme } from '../components/ThemeProvider';
import { Server, Cpu, HardDrive, Network, Zap } from 'lucide-react';

export default function Infrastructure() {
  const { theme } = useTheme();

  const textColor = theme === 'dark' ? '#94a3b8' : '#64748b';
  const gridColor = theme === 'dark' ? '#1e293b' : '#e2e8f0';

  // Mock data for multi-GPU Utilization
  const gpuUtilOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      grid: { left: 40, right: 20, bottom: 30, top: 20 },
      xAxis: { type: 'category', data: Array.from({length: 20}, (_, i) => `Step ${i*100}`), axisLabel: { color: textColor } },
      yAxis: { type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      series: [
        { name: 'GPU 0', type: 'line', data: Array.from({length: 20}, () => 85 + Math.random()*15), smooth: true, itemStyle: { color: '#3b82f6' }, symbol: 'none' },
        { name: 'GPU 1', type: 'line', data: Array.from({length: 20}, () => 80 + Math.random()*20), smooth: true, itemStyle: { color: '#10b981' }, symbol: 'none' },
        { name: 'GPU 2', type: 'line', data: Array.from({length: 20}, () => 88 + Math.random()*10), smooth: true, itemStyle: { color: '#8b5cf6' }, symbol: 'none' },
        { name: 'GPU 3', type: 'line', data: Array.from({length: 20}, () => 82 + Math.random()*18), smooth: true, itemStyle: { color: '#f59e0b' }, symbol: 'none' },
      ]
    };
  }, [textColor, gridColor]);

  // Mock data for GPU Memory
  const gpuMemOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 50, right: 30, bottom: 40, top: 10 },
      xAxis: { type: 'value', max: 80, name: 'Memory (GB)', nameLocation: 'middle', nameGap: 25, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      yAxis: { type: 'category', data: ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3'], axisLabel: { color: textColor } },
      series: [
        { name: 'Allocated', type: 'bar', stack: 'total', data: [72.4, 73.1, 72.8, 72.9], itemStyle: { color: '#ef4444' } },
        { name: 'Reserved', type: 'bar', stack: 'total', data: [2.1, 1.8, 2.0, 1.9], itemStyle: { color: '#f59e0b' } },
        { name: 'Free', type: 'bar', stack: 'total', data: [5.5, 5.1, 5.2, 5.2], itemStyle: { color: theme === 'dark' ? '#334155' : '#e2e8f0' } }
      ]
    };
  }, [theme, textColor, gridColor]);

  // Mock data for Network traffic (NCCL)
  const networkOptions = useMemo(() => {
    return {
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      grid: { left: 50, right: 20, bottom: 30, top: 10 },
      xAxis: { type: 'category', data: Array.from({length: 20}, (_, i) => `${i*10}s`), axisLabel: { color: textColor } },
      yAxis: { type: 'value', name: 'GB/s', nameLocation: 'end', splitLine: { lineStyle: { color: gridColor, type: 'dashed' } }, axisLabel: { color: textColor } },
      series: [
        { name: 'NCCL Sync (AllReduce)', type: 'line', data: Array.from({length: 20}, () => 40 + Math.random()*20), smooth: true, itemStyle: { color: '#06b6d4' }, areaStyle: { opacity: 0.2, color: '#06b6d4' }, symbol: 'none' }
      ]
    };
  }, [textColor, gridColor]);

  const stats = [
    { label: 'Cluster Nodes', val: '1 / 1', icon: Server, color: 'text-indigo-500' },
    { label: 'System CPU', val: '34%', icon: Cpu, color: 'text-emerald-500' },
    { label: 'System RAM', val: '124 / 256 GB', icon: HardDrive, color: 'text-violet-500' },
    { label: 'Power Draw', val: '1,420 W', icon: Zap, color: 'text-amber-500' },
  ];

  return (
    <div className="space-y-6 flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
           <h1 className="text-2xl font-bold text-slate-900 dark:text-white tracking-tight">Infrastructure Telemetry</h1>
           <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Real-time hardware utilization, multi-GPU synchronization, and memory health.</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
         {stats.map((s, i) => {
            const Icon = s.icon;
            return (
              <div key={i} className="card flex flex-col justify-between">
                <div className="flex items-center justify-between mb-2 opacity-80">
                  <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{s.label}</p>
                  <Icon className={`w-4 h-4 ${s.color}`} />
                </div>
                <p className="text-3xl font-bold text-slate-900 dark:text-white">{s.val}</p>
              </div>
            )
         })}
      </div>

      <div className="flex-1 overflow-y-auto custom-scrollbar space-y-6 pr-2">
         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="card h-[380px] flex flex-col">
               <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">Multi-GPU Compute Utilization</h3>
               <div className="flex-1 min-h-0">
                  <ReactECharts option={gpuUtilOptions} style={{ height: '100%', width: '100%' }} />
               </div>
            </div>
            
            <div className="card h-[380px] flex flex-col">
               <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2">GPU Memory Fragmentation</h3>
               <div className="flex-1 min-h-0">
                  <ReactECharts option={gpuMemOptions} style={{ height: '100%', width: '100%' }} />
               </div>
            </div>
         </div>

         <div className="card h-[300px] flex flex-col">
            <h3 className="text-base font-semibold text-slate-900 dark:text-white mb-2 flex items-center gap-2">
              <Network className="w-5 h-5 text-cyan-500" />
              NCCL Ring / Network Bandwidth
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">Tracking AllReduce synchronization stalls across PCIe / NVLink bridges.</p>
            <div className="flex-1 min-h-0">
               <ReactECharts option={networkOptions} style={{ height: '100%', width: '100%' }} />
            </div>
         </div>
      </div>
    </div>
  );
}
