import React, { useEffect, useState, useMemo } from 'react';
import { useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import ReactECharts from 'echarts-for-react';
import { Settings2, Eye, EyeOff } from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';

export default function Training() {
  const { runId } = useParams();
  const { setActiveRun, metrics, discoverMetricKeys } = useRunStore();
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) setActiveRun(runId);
  }, [runId, setActiveRun]);

  const availableMetrics = useMemo(() => {
    const keys = discoverMetricKeys();
    return keys.filter((key) =>
      metrics.some((m) => {
        const value = m[key];
        if (value == null) return false;
        const numeric = typeof value === 'number' ? value : Number(value);
        return Number.isFinite(numeric);
      })
    );
  }, [metrics, discoverMetricKeys]);

  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);

  useEffect(() => {
    setSelectedMetrics([]);
    setHasUserInteracted(false);
  }, [runId]);

  useEffect(() => {
    const availableSet = new Set(availableMetrics);
    setSelectedMetrics((prev) => {
      const filtered = prev.filter((key) => availableSet.has(key));
      if (!hasUserInteracted && filtered.length === 0 && availableMetrics.length > 0) {
        return availableMetrics.slice(0, 3);
      }
      return filtered;
    });
  }, [availableMetrics, hasUserInteracted]);

  const [smoothing, setSmoothing] = useState(0.2); // 0 to 0.99
  
  const toggleMetric = (key) => {
    setHasUserInteracted(true);
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
    const legendMetrics = [];
    const colors = ['#38bdf8', '#a78bfa', '#fb7185', '#34d399', '#fbbf24', '#f472b6', '#818cf8'];
    const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
    const mutedTextColor = theme === 'dark' ? '#94a3b8' : '#64748b';
    const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.08)';
    const tooltipBackground = theme === 'dark' ? '#1e293b' : '#ffffff';
    const tooltipBorder = theme === 'dark' ? '#334155' : '#d1d5db';
    const tooltipText = theme === 'dark' ? '#f8fafc' : '#0f172a';

    selectedMetrics.forEach((metricKey, i) => {
      const color = colors[i % colors.length];
      const rawData = metrics
        .map((m) => {
          if (!(metricKey in m)) return null;
          const stepValue = typeof m.step === 'number' ? m.step : Number(m.step);
          const metricValue = typeof m[metricKey] === 'number' ? m[metricKey] : Number(m[metricKey]);
          if (!Number.isFinite(stepValue) || !Number.isFinite(metricValue)) return null;
          return [stepValue, metricValue];
        })
        .filter(Boolean);
      if (rawData.length === 0) return;

      const smoothedData = getSmoothedData(rawData, smoothing);
      legendMetrics.push(metricKey);
      
      // Add raw (faded) and smoothed series
      if (smoothing > 0) {
        series.push({
          id: `raw:${metricKey}`,
          name: `${metricKey} (raw)`,
          type: 'line',
          data: rawData,
          showSymbol: false,
          lineStyle: { color, opacity: 0.2, width: 1 },
          itemStyle: { color }
        });
      }

      series.push({
        id: `smooth:${metricKey}`,
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
        backgroundColor: tooltipBackground,
        borderColor: tooltipBorder,
        textStyle: { color: tooltipText },
        axisPointer: { type: 'cross', label: { backgroundColor: tooltipBorder, color: tooltipText } }
      },
      legend: {
        data: legendMetrics,
        textStyle: { color: textColor },
         top: 0
      },
      grid: { left: 60, right: 60, bottom: 88, top: 40, containLabel: false },
      toolbox: {
        feature: {
          dataZoom: { yAxisIndex: 'none' },
          restore: {},
        },
        iconStyle: { borderColor: textColor }
      },
      dataZoom: [
        { type: 'inside', start: 0, end: 100 },
        {
          type: 'slider',
          start: 0,
          end: 100,
          bottom: 6,
          height: 22,
          textStyle: { color: mutedTextColor },
          borderColor: gridColor,
          fillerColor: theme === 'dark' ? 'rgba(56, 189, 248, 0.16)' : 'rgba(56, 189, 248, 0.18)',
          backgroundColor: theme === 'dark' ? 'rgba(148, 163, 184, 0.14)' : 'rgba(100, 116, 139, 0.12)',
        }
      ],
      xAxis: {
        type: 'value',
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 45,
        splitLine: { show: false },
        axisLabel: { color: mutedTextColor },
        axisLine: { lineStyle: { color: gridColor } }
      },
      yAxis: {
        type: 'value',
        name: 'Value',
        splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
        axisLabel: { color: mutedTextColor },
        axisLine: { lineStyle: { color: gridColor } }
      },
      series
    };
  }, [metrics, selectedMetrics, smoothing, theme]);

  if (availableMetrics.length === 0) {
    return <div className="p-8 text-theme-text-secondary">No scalar metrics found matching this run.</div>;
  }

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between mb-6 shrink-0">
        <div>
          <h1 className="text-2xl font-bold text-theme-text-primary tracking-tight">Training Metrics</h1>
          <p className="text-sm text-theme-text-secondary mt-1">Interactive exploration of logged scalars.</p>
        </div>
        
        <div className="flex items-center gap-4 bg-theme-surface border border-theme-border rounded-lg py-2 px-4 shadow-sm">
          <Settings2 className="w-4 h-4 text-theme-text-muted" />
          <div className="flex items-center gap-3">
            <span className="text-sm text-theme-text-secondary font-medium">Smoothing</span>
            <input 
              type="range" 
              min="0" max="0.99" step="0.01" 
              value={smoothing} 
              onChange={(e) => setSmoothing(parseFloat(e.target.value))}
              className="w-32 accent-indigo-500"
            />
            <span className="text-xs text-theme-text-muted font-mono w-8 text-right">{smoothing.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="flex gap-6 flex-1 min-h-0">
        <div className="w-64 shrink-0 overflow-y-auto pr-2 custom-scrollbar">
          <h3 className="text-xs font-semibold text-theme-text-muted uppercase tracking-wider mb-3">Available Metrics</h3>
          <div className="space-y-1">
            {availableMetrics.map(key => {
              const isSelected = selectedMetrics.includes(key);
              return (
                <button
                  key={key}
                  onClick={() => toggleMetric(key)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-sm transition-all border ${
                    isSelected 
                      ? 'bg-theme-primary/10 border-theme-primary/40 text-theme-primary' 
                      : 'bg-theme-surface border-theme-border text-theme-text-secondary hover:bg-theme-surface-hover hover:text-theme-text-primary'
                  }`}
                >
                  <span className="truncate" title={key}>{key}</span>
                  {isSelected ? <Eye className="w-4 h-4 shrink-0 ml-2" /> : <EyeOff className="w-4 h-4 shrink-0 ml-2 opacity-50" />}
                </button>
              );
            })}
          </div>
        </div>
        
        <div className="card flex-1 min-w-0">
          {selectedMetrics.length === 0 ? (
            <div className="h-full flex items-center justify-center text-theme-text-muted">
              Select metrics from the sidebar to view them here.
            </div>
          ) : (
            <ReactECharts 
              option={chartOptions} 
              style={{ height: '100%', width: '100%' }}
              notMerge={true}
              replaceMerge={['series', 'legend']}
              lazyUpdate={true}
            />
          )}
        </div>
      </div>
    </div>
  );
}
