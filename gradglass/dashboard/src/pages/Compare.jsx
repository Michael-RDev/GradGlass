import React, { useEffect, useState, useMemo } from 'react';
import { fetchRuns, fetchCompare } from '../api';
import ReactECharts from 'echarts-for-react';
import { GitCompare, Filter, CheckSquare, Square, Search } from 'lucide-react';

export default function Compare() {
  const [runs, setRuns] = useState([]);
  const [selectedRunIds, setSelectedRunIds] = useState([]);
  const [compareData, setCompareData] = useState({});
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  // 1. Fetch all available runs
  useEffect(() => {
    fetchRuns().then(res => setRuns(res.runs || [])).catch(console.error);
  }, []);

  // 2. Fetch compare data when selection changes
  useEffect(() => {
    if (selectedRunIds.length === 0) {
      setCompareData({});
      return;
    }
    setLoading(true);
    fetchCompare(selectedRunIds)
      .then(res => setCompareData(res))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedRunIds]);

  const toggleRun = (id) => {
    setSelectedRunIds(prev =>
      prev.includes(id) ? prev.filter(r => r !== id) : [...prev, id]
    );
  };

  const filteredRuns = useMemo(() => {
    if (!searchQuery) return runs;
    return runs.filter(r => 
      r.run_id.toLowerCase().includes(searchQuery.toLowerCase()) || 
      (r.name && r.name.toLowerCase().includes(searchQuery.toLowerCase()))
    );
  }, [runs, searchQuery]);

  // Discover common string metrics across selected runs
  const availableMetrics = useMemo(() => {
    const keys = new Set();
    Object.values(compareData).forEach(runData => {
      runData.metrics?.forEach(m => {
        Object.keys(m).forEach(k => {
          if (k !== 'step' && k !== 'timestamp') keys.add(k);
        });
      });
    });
    return Array.from(keys).sort();
  }, [compareData]);

  const [selectedMetric, setSelectedMetric] = useState('loss');
  useEffect(() => {
    if (availableMetrics.length > 0 && !availableMetrics.includes(selectedMetric)) {
      setSelectedMetric(availableMetrics[0]);
    }
  }, [availableMetrics, selectedMetric]);

  const chartOptions = useMemo(() => {
    const series = [];
    const colors = ['#38bdf8', '#fb7185', '#a78bfa', '#34d399', '#fbbf24', '#f472b6', '#818cf8'];

    selectedRunIds.forEach((rid, i) => {
      const runData = compareData[rid];
      if (!runData || !runData.metrics) return;
      const color = colors[i % colors.length];
      
      const rawData = runData.metrics
        .filter(m => selectedMetric in m && m[selectedMetric] !== null)
        .map(m => [m.step, m[selectedMetric]]);
      
      if (rawData.length === 0) return;

      series.push({
        name: runs.find(r => r.run_id === rid)?.name || rid,
        type: 'line',
        data: rawData,
        showSymbol: false,
        smooth: true,
        lineStyle: { color, width: 2 },
        itemStyle: { color }
      });
    });

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' }
      },
      legend: {
        textStyle: { color: '#cbd5e1' },
      },
      grid: { left: 60, right: 30, bottom: 40, top: 40, containLabel: false },
      toolbox: {
        feature: { dataZoom: { yAxisIndex: 'none' }, restore: {} },
        iconStyle: { borderColor: '#94a3b8' }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100, textStyle: { color: '#94a3b8' } }
      ],
      xAxis: { type: 'value', name: 'Step', nameLocation: 'middle', nameGap: 25, splitLine: { show: false }, axisLabel: { color: '#94a3b8' } },
      yAxis: { type: 'value', name: selectedMetric, splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } }, axisLabel: { color: '#94a3b8' } },
      series
    };
  }, [compareData, selectedRunIds, selectedMetric, runs]);

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-6 shrink-0">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Compare Experiments</h1>
          <p className="text-sm text-slate-400 mt-1">Overlay metrics from multiple runs to analyze performance.</p>
        </div>

        {availableMetrics.length > 0 && (
          <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 rounded-lg py-2 px-4 shadow-sm">
            <Filter className="w-4 h-4 text-slate-400" />
            <select 
              className="bg-slate-950 border border-slate-700 text-sm text-slate-300 rounded-md py-1 px-3 outline-none focus:ring-1 focus:ring-indigo-500"
              value={selectedMetric}
              onChange={e => setSelectedMetric(e.target.value)}
            >
              {availableMetrics.map(m => (
                 <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        {/* Left Sidebar: Run Selection */}
        <div className="w-80 shrink-0 flex flex-col bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
           <div className="p-4 border-b border-slate-800">
              <div className="relative">
                 <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                 <input 
                   type="text" 
                   value={searchQuery}
                   onChange={e => setSearchQuery(e.target.value)}
                   placeholder="Filter runs..." 
                   className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-9 pr-3 py-2 text-sm text-slate-200 outline-none focus:border-indigo-500"
                 />
              </div>
           </div>
           
           <div className="flex-1 overflow-y-auto custom-scrollbar p-2">
              {filteredRuns.length === 0 ? (
                <div className="p-4 text-sm text-slate-500 text-center">No runs found.</div>
              ) : (
                filteredRuns.map(run => {
                  const isSelected = selectedRunIds.includes(run.run_id);
                  return (
                    <button
                      key={run.run_id}
                      onClick={() => toggleRun(run.run_id)}
                      className={`w-full flex items-start gap-3 p-3 mb-1 rounded-lg text-left transition-colors border ${
                        isSelected 
                          ? 'bg-indigo-500/10 border-indigo-500/30' 
                          : 'bg-transparent border-transparent hover:bg-slate-800/50 hover:border-slate-700'
                      }`}
                    >
                      <div className="mt-0.5">
                        {isSelected ? <CheckSquare className="w-4 h-4 text-indigo-400" /> : <Square className="w-4 h-4 text-slate-500" />}
                      </div>
                      <div className="min-w-0">
                         <p className={`text-sm font-medium truncate ${isSelected ? 'text-indigo-300' : 'text-slate-300'}`}>
                           {run.name || run.run_id}
                         </p>
                         <div className="flex flex-wrap gap-2 mt-1">
                            <span className="text-xs font-mono text-slate-500 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                               {run.run_id.slice(-6)}
                            </span>
                            {run.latest_acc != null && (
                               <span className="text-xs text-slate-400">acc: {(run.latest_acc*100).toFixed(1)}%</span>
                            )}
                         </div>
                      </div>
                    </button>
                  );
                })
              )}
           </div>
        </div>

        {/* Right Area: Chart */}
        <div className="flex-1 bg-slate-900 shadow-xl border border-slate-800 rounded-xl p-6 min-w-0 flex flex-col">
          {selectedRunIds.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
              <GitCompare className="w-12 h-12 text-slate-700 mb-4" />
              <p>Select multiple runs from the sidebar to compare them.</p>
            </div>
          ) : loading && Object.keys(compareData).length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-slate-400">
              Loading comparison data...
            </div>
          ) : availableMetrics.length === 0 ? (
            <div className="flex-1 flex flex-col items-center justify-center text-slate-500">
              <p>No common numerical metrics found across these runs.</p>
            </div>
          ) : (
            <>
              <h3 className="text-base font-semibold text-white mb-4 flex items-center gap-2">
                Comparison: {selectedMetric}
              </h3>
              <div className="flex-1 min-h-0">
                <ReactECharts 
                  option={chartOptions} 
                  style={{ height: '100%', width: '100%' }}
                  notMerge={true}
                />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
