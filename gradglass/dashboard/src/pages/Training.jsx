import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import ReactECharts from 'echarts-for-react';
import { Settings2, Minus, Eye, EyeOff } from 'lucide-react';

export default function Training() {
  const { runId } = useParams();
  const { setActiveRun, metrics, discoverMetricKeys } = useRunStore();

  useEffect(() => {
    if (runId) setActiveRun(runId);
  }, [runId, setActiveRun]);

  const availableMetrics = useMemo(() => discoverMetricKeys(), [metrics, discoverMetricKeys]);
  
  // By default, select the first 3 metrics if available
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  useEffect(() => {
    if (availableMetrics.length > 0 && selectedMetrics.length === 0) {
      setSelectedMetrics(availableMetrics.slice(0, 3));
    }
  }, [availableMetrics]);

  const [smoothing, setSmoothing] = useState(0.2); // 0 to 0.99
  
  const toggleMetric = (key) => {
    setSelectedMetrics(prev => 
      prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]
    );
  };

  // Exponential moving average for smoothing
  const getSmoothedData = (data, weight) => {
    if (weight === 0) return data;
    let last = data[0][1];
    return data.map(([x, y]) => {
      if (y == null) return [x, null];
      last = last * weight + y * (1 - weight);
      return [x, last];
    });
  };

  const chartOptions = useMemo(() => {
    const series = [];
    const colors = ['#38bdf8', '#a78bfa', '#fb7185', '#34d399', '#fbbf24', '#f472b6', '#818cf8'];

    selectedMetrics.forEach((metricKey, i) => {
      const color = colors[i % colors.length];
      const rawData = metrics.filter(m => metricKey in m && m[metricKey] !== null).map(m => [m.step, m[metricKey]]);
      if (rawData.length === 0) return;

      const smoothedData = getSmoothedData(rawData, smoothing);
      
      // Add raw (faded) and smoothed series
      if (smoothing > 0) {
        series.push({
          name: `${metricKey} (raw)`,
          type: 'line',
          data: rawData,
          showSymbol: false,
          lineStyle: { color, opacity: 0.2, width: 1 },
          itemStyle: { color }
        });
      }

      series.push({
        name: metricKey,
        type: 'line',
        data: smoothing > 0 ? smoothedData : rawData,
        showSymbol: false,
        smooth: false,
        lineStyle: { color, width: 2 },
        itemStyle: { color }
      });
    });

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1e293b',
        borderColor: '#334155',
        textStyle: { color: '#f8fafc' },
        axisPointer: { type: 'cross', label: { backgroundColor: '#334155' } }
      },
      legend: {
        data: selectedMetrics,
        textStyle: { color: '#cbd5e1' },
         top: 0
      },
      grid: { left: 60, right: 60, bottom: 40, top: 40, containLabel: false },
      toolbox: {
        feature: {
          dataZoom: { yAxisIndex: 'none' },
          restore: {},
        },
        iconStyle: { borderColor: '#94a3b8' }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        { type: 'slider', start: 0, end: 100, textStyle: { color: '#94a3b8' } }
      ],
      xAxis: {
        type: 'value',
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 25,
        splitLine: { show: false },
        axisLabel: { color: '#94a3b8' }
      },
      yAxis: {
        type: 'value',
        name: 'Value',
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
        axisLabel: { color: '#94a3b8' }
      },
      series
    };
  }, [metrics, selectedMetrics, smoothing]);

  if (availableMetrics.length === 0) {
    return <div className="p-8 text-slate-400">No scalar metrics found matching this run.</div>;
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-6 shrink-0">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Training Metrics</h1>
          <p className="text-sm text-slate-400 mt-1">Interactive exploration of logged scalars.</p>
        </div>
        
        <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 rounded-lg py-2 px-4 shadow-sm">
          <Settings2 className="w-4 h-4 text-slate-400" />
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-300 font-medium">Smoothing</span>
            <input 
              type="range" 
              min="0" max="0.99" step="0.01" 
              value={smoothing} 
              onChange={(e) => setSmoothing(parseFloat(e.target.value))}
              className="w-32 accent-indigo-500"
            />
            <span className="text-xs text-slate-400 font-mono w-8 text-right">{smoothing.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        <div className="w-64 shrink-0 overflow-y-auto pr-2 custom-scrollbar">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">Available Metrics</h3>
          <div className="space-y-1">
            {availableMetrics.map(key => {
              const isSelected = selectedMetrics.includes(key);
              return (
                <button
                  key={key}
                  onClick={() => toggleMetric(key)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-sm transition-all border ${
                    isSelected 
                      ? 'bg-indigo-500/10 border-indigo-500/30 text-indigo-400' 
                      : 'bg-slate-900 border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-slate-300'
                  }`}
                >
                  <span className="truncate" title={key}>{key}</span>
                  {isSelected ? <Eye className="w-4 h-4 shrink-0 ml-2" /> : <EyeOff className="w-4 h-4 shrink-0 ml-2 opacity-50" />}
                </button>
              );
            })}
          </div>
        </div>
        
        <div className="flex-1 bg-slate-900 shadow-xl border border-slate-800 rounded-xl p-6 min-w-0">
          {selectedMetrics.length === 0 ? (
            <div className="h-full flex items-center justify-center text-slate-500">
              Select metrics from the sidebar to view them here.
            </div>
          ) : (
            <ReactECharts 
              option={chartOptions} 
              style={{ height: '100%', width: '100%' }}
              notMerge={false}
              lazyUpdate={true}
            />
          )}
        </div>
      </div>
    </div>
  );
}
