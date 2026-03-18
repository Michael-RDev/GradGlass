import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import { Activity, BarChart3, Cpu, Database, Gauge, HardDrive, Network, Server, Thermometer, Zap } from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';
import { fetchInfrastructureTelemetry } from '../api';
import { StatusBadge } from '../components/ui';

const POLL_INTERVAL_MS = 2000;
const HISTORY_LIMIT = 60;
const SERIES_COLORS = ['#FDA481', '#B4182D', '#54162B', '#37415C', '#2F8F9D', '#6A7286', '#22C55E', '#0EA5E9'];

const STATUS_STYLES = {
  active: 'text-emerald-700 bg-emerald-500/10 border-emerald-500/30 dark:text-emerald-300',
  not_detected: 'text-slate-700 bg-slate-500/10 border-slate-500/30 dark:text-slate-300',
  not_supported: 'text-amber-700 bg-amber-500/10 border-amber-500/30 dark:text-amber-300',
  disabled_local_mode: 'text-slate-700 bg-slate-500/10 border-slate-500/30 dark:text-slate-300',
  requires_cluster_connection: 'text-indigo-700 bg-indigo-500/10 border-indigo-500/30 dark:text-indigo-300',
  dependency_missing: 'text-rose-700 bg-rose-500/10 border-rose-500/30 dark:text-rose-300',
  interrupted_training_stopped: 'text-orange-700 bg-orange-500/10 border-orange-500/30 dark:text-orange-300',
  error: 'text-red-700 bg-red-500/10 border-red-500/30 dark:text-red-300',
};

function formatTimestamp(ts) {
  if (!ts) return '—';
  const dt = new Date(Number(ts) * 1000);
  if (Number.isNaN(dt.getTime())) return '—';
  return dt.toLocaleTimeString();
}

function formatNumber(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function modeLabel(mode) {
  if (mode === 'cuda_active') return 'CUDA active';
  if (mode === 'mps_active') return 'MPS active';
  if (mode === 'cpu_only') return 'CPU-only mode';
  if (mode === 'heterogeneous_active') return 'Multi-accelerator active';
  return 'Accelerator unavailable';
}

function modeStyle(mode) {
  if (mode === 'cuda_active') return 'text-emerald-700 bg-emerald-500/10 border-emerald-500/30 dark:text-emerald-300';
  if (mode === 'mps_active') return 'text-sky-700 bg-sky-500/10 border-sky-500/30 dark:text-sky-300';
  if (mode === 'cpu_only') return 'text-slate-700 bg-slate-500/10 border-slate-500/30 dark:text-slate-300';
  if (mode === 'heterogeneous_active') return 'text-indigo-700 bg-indigo-500/10 border-indigo-500/30 dark:text-indigo-300';
  return 'text-rose-700 bg-rose-500/10 border-rose-500/30 dark:text-rose-300';
}

function metricStatusClass(status) {
  return STATUS_STYLES[status] || STATUS_STYLES.error;
}

function getMetric(group, key) {
  if (!group || typeof group !== 'object') return null;
  const metric = group[key];
  return metric && typeof metric === 'object' ? metric : null;
}

function metricValue(metric) {
  if (!metric || typeof metric !== 'object') return null;
  const value = Number(metric.value);
  return Number.isFinite(value) ? value : null;
}

function hasFiniteValue(seriesValues) {
  return Array.isArray(seriesValues) && seriesValues.some((value) => Number.isFinite(value));
}

function MetricBadge({ metric }) {
  const status = metric?.status || 'error';
  const label = metric?.status_label || 'Unknown';
  return <span className={`badge ${metricStatusClass(status)}`}>{label}</span>;
}

function MetricTile({ title, icon: Icon, metric, fallback = '—' }) {
  const valueText = metric?.display || fallback;
  return (
    <div className="card">
      <div className="flex items-center justify-between opacity-90">
        <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{title}</p>
        <Icon className="h-4 w-4 text-theme-accent" />
      </div>
      <p className="mt-3 text-2xl font-bold text-slate-900 dark:text-white">{valueText}</p>
      <div className="mt-2 flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
        <MetricBadge metric={metric} />
        <span>Updated {formatTimestamp(metric?.timestamp)}</span>
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
        Source: <span className="font-mono">{metric?.source || 'unknown'}</span>
      </p>
      <p className="text-xs text-slate-500 dark:text-slate-400 break-all">
        Probe: <span className="font-mono">{metric?.probe || 'unknown'}</span>
      </p>
    </div>
  );
}

function AcceleratorCard({ accelerator, historySeries, gridColor }) {
  const util = getMetric(accelerator?.metrics, 'utilization_percent');
  const mem = getMetric(accelerator?.metrics, 'memory_pressure_percent');
  const frag = getMetric(accelerator?.metrics, 'memory_fragmentation_percent');
  const power = getMetric(accelerator?.metrics, 'power_watts');
  const fan = getMetric(accelerator?.metrics, 'fan_speed_percent');

  const miniOption = {
    tooltip: { trigger: 'axis' },
    grid: { left: 8, right: 8, top: 10, bottom: 10 },
    xAxis: { type: 'category', show: false, data: historySeries.labels },
    yAxis: {
      type: 'value',
      min: 0,
      max: 100,
      show: false,
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    series: [
      {
        type: 'line',
        data: historySeries.values,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#FDA481' },
        areaStyle: { color: 'rgba(253, 164, 129, 0.16)' },
      },
    ],
  };

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-semibold text-theme-text-primary">{accelerator?.name || accelerator?.id || 'Accelerator'}</h3>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            {accelerator?.backend?.toUpperCase() || 'ACCELERATOR'} · {accelerator?.vendor || 'unknown vendor'}
          </p>
        </div>
        <MetricBadge metric={util || { status: accelerator?.status || 'not_detected', status_label: 'Status unavailable' }} />
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        <p className="text-slate-500 dark:text-slate-400">Utilization</p>
        <p className="text-right font-mono text-theme-text-primary">{util?.display || '—'}</p>
        <p className="text-slate-500 dark:text-slate-400">Memory pressure</p>
        <p className="text-right font-mono text-theme-text-primary">{mem?.display || '—'}</p>
        <p className="text-slate-500 dark:text-slate-400">Fragmentation</p>
        <p className="text-right font-mono text-theme-text-primary">{frag?.display || '—'}</p>
        <p className="text-slate-500 dark:text-slate-400">Power</p>
        <p className="text-right font-mono text-theme-text-primary">{power?.display || '—'}</p>
        <p className="text-slate-500 dark:text-slate-400">Fan</p>
        <p className="text-right font-mono text-theme-text-primary">{fan?.display || '—'}</p>
      </div>
      <div className="mt-3 h-20">
        {historySeries.values.length > 1 ? (
          <ReactECharts option={miniOption} style={{ height: '100%', width: '100%' }} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-slate-400">
            Waiting for utilization samples...
          </div>
        )}
      </div>
      <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Updated {formatTimestamp(util?.timestamp || mem?.timestamp)}</p>
    </div>
  );
}

function buildLineOptions({ labels, series, yAxisName, yAxisMax, textColor, gridColor, dualAxis = false }) {
  const normalizedSeries = (series || []).map((s) => {
    if (!s || typeof s !== 'object') return s;
    const data = Array.isArray(s.data) ? s.data : [];
    const finiteCount = data.reduce((acc, value) => (Number.isFinite(value) ? acc + 1 : acc), 0);
    if (finiteCount <= 1) {
      return { ...s, showSymbol: true, symbolSize: 7 };
    }
    return s;
  });
  return {
    animationDuration: 260,
    animationDurationUpdate: 220,
    tooltip: { trigger: 'axis' },
    legend: { textStyle: { color: textColor }, top: 0 },
    grid: { left: 48, right: dualAxis ? 54 : 24, bottom: 36, top: 36 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: textColor },
      axisLine: { lineStyle: { color: gridColor } },
    },
    yAxis: dualAxis
      ? [
          {
            type: 'value',
            min: 0,
            max: 100,
            axisLabel: { color: textColor, formatter: '{value}%' },
            name: yAxisName,
            nameTextStyle: { color: textColor },
            splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
          },
          {
            type: 'value',
            min: 0,
            axisLabel: { color: textColor },
            name: 'Power (W)',
            nameTextStyle: { color: textColor },
            splitLine: { show: false },
          },
        ]
      : {
          type: 'value',
          min: 0,
          max: yAxisMax,
          axisLabel: { color: textColor },
          name: yAxisName,
          nameTextStyle: { color: textColor },
          splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
        },
    series: normalizedSeries,
  };
}

export default function Infrastructure() {
  const { runId } = useParams();
  const { theme } = useTheme();
  const [telemetry, setTelemetry] = useState(null);
  const [history, setHistory] = useState([]);
  const [fetchError, setFetchError] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!runId) return undefined;

    let active = true;
    let timer = null;

    const poll = async () => {
      try {
        const payload = await fetchInfrastructureTelemetry(runId);
        if (!active) return;

        const v2 = payload?.telemetry_v2 || {};
        const aggregate = v2?.aggregate_accelerator?.metrics || {};
        const host = v2?.host_metrics || {};
        const process = v2?.training_process_metrics || {};
        const cluster = v2?.cluster_metrics || {};
        const external = v2?.external_usage || {};
        const local = v2?.local_performance_insights || {};
        const graphHints = v2?.graph_hints || {};

        const ts = Number(payload?.collected_at) || Date.now() / 1000;

        const perAccelUtil = {};
        const perAccelMem = {};
        const accelerators = Array.isArray(v2?.accelerators) ? v2.accelerators : [];
        accelerators.forEach((acc, idx) => {
          const name = acc?.name || acc?.id || `Accelerator ${idx}`;
          const utilValue = metricValue(getMetric(acc?.metrics, 'utilization_percent'));
          const memValue = metricValue(getMetric(acc?.metrics, 'memory_pressure_percent'));
          if (utilValue != null) perAccelUtil[name] = utilValue;
          if (memValue != null) perAccelMem[name] = memValue;
        });

        const point = {
          ts,
          aggregateUtil: metricValue(getMetric(aggregate, 'utilization_percent')),
          aggregateMem: metricValue(getMetric(aggregate, 'memory_pressure_percent')),
          aggregateFrag: metricValue(getMetric(aggregate, 'memory_fragmentation_percent')),
          aggregatePower: metricValue(getMetric(aggregate, 'power_watts')),
          aggregateTemp: metricValue(getMetric(aggregate, 'temperature_c')),
          aggregateInterconnectRx: metricValue(getMetric(aggregate, 'interconnect_rx_mb_s')),
          hostCpu:
            metricValue(getMetric(external, 'host_cpu_percent')) ??
            metricValue(getMetric(host, 'system_cpu_percent')),
          processCpu:
            metricValue(getMetric(external, 'process_cpu_percent')) ??
            metricValue(getMetric(process, 'process_cpu_percent')),
          processRam:
            metricValue(getMetric(external, 'process_ram_percent')) ??
            metricValue(getMetric(process, 'process_ram_percent')),
          ramPressure:
            metricValue(getMetric(external, 'host_ram_percent')) ??
            metricValue(getMetric(host, 'system_ram_percent')),
          diskRead:
            metricValue(getMetric(external, 'disk_read_mb_s')) ??
            metricValue(getMetric(host, 'disk_read_mb_s')),
          diskWrite:
            metricValue(getMetric(external, 'disk_write_mb_s')) ??
            metricValue(getMetric(host, 'disk_write_mb_s')),
          netRx:
            metricValue(getMetric(external, 'net_rx_mb_s')) ??
            metricValue(getMetric(host, 'network_rx_mb_s')),
          netTx:
            metricValue(getMetric(external, 'net_tx_mb_s')) ??
            metricValue(getMetric(host, 'network_tx_mb_s')),
          samplesPerSec: metricValue(getMetric(local, 'samples_per_sec')),
          dataloaderWait: metricValue(getMetric(local, 'dataloader_wait_time_s')),
          hostToDeviceTransfer: metricValue(getMetric(local, 'host_to_device_transfer_time_s')),
          cpuBottleneck: metricValue(getMetric(local, 'cpu_bottleneck_score')),
          memoryPressureScore: metricValue(getMetric(local, 'memory_pressure')),
          diskPressureScore: metricValue(getMetric(local, 'disk_io_pressure')),
          acceleratorStarvation: metricValue(getMetric(local, 'accelerator_starvation_idle')),
          scalingReadiness: metricValue(getMetric(local, 'scaling_readiness')),
          clusterRx: metricValue(getMetric(cluster, 'interconnect_rx_mb_s')),
          clusterTx: metricValue(getMetric(cluster, 'interconnect_tx_mb_s')),
          perAccelUtil,
          perAccelMem,
        };

        setTelemetry(payload);
        setFetchError(null);
        setIsLoading(false);
        setHistory((prev) => {
          const stabilized = { ...point };
          const last = prev.length > 0 ? prev[prev.length - 1] : null;
          const runTerminal = Boolean(graphHints?.run_terminal);
          if (runTerminal && last) {
            ['processCpu', 'processRam', 'samplesPerSec', 'dataloaderWait', 'hostToDeviceTransfer'].forEach((key) => {
              if (!Number.isFinite(stabilized[key]) && Number.isFinite(last[key])) {
                stabilized[key] = last[key];
              }
            });
          }
          return [...prev, stabilized].slice(-HISTORY_LIMIT);
        });
      } catch (err) {
        if (!active) return;
        setFetchError(err?.message || 'Failed to fetch infrastructure telemetry');
        setIsLoading(false);
      } finally {
        if (active) timer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [runId]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.08)';

  const v2 = telemetry?.telemetry_v2 || {};
  const aggregate = v2?.aggregate_accelerator?.metrics || {};
  const host = v2?.host_metrics || {};
  const processMetrics = v2?.training_process_metrics || {};
  const externalUsage = v2?.external_usage || {};
  const graphHints = v2?.graph_hints || {};
  const localInsights = v2?.local_performance_insights || {};
  const clusterMetrics = v2?.cluster_metrics || {};
  const runState = v2?.run_state || {};
  const accelerators = Array.isArray(v2?.accelerators) ? v2.accelerators : [];
  const preferredLayout = graphHints?.preferred_layout || 'host_process_first';
  const runTerminal = Boolean(graphHints?.run_terminal);
  const throughputWarmup = Boolean(graphHints?.throughput_warmup);

  const labels = history.map((entry) => formatTimestamp(entry.ts));

  const utilizationSeries = useMemo(() => {
    const aggregateLine = {
      name: 'Aggregate utilization',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.aggregateUtil) ? entry.aggregateUtil : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 3, color: '#FDA481' },
      areaStyle: { color: 'rgba(253, 164, 129, 0.12)' },
    };

    const perAcceleratorLines = [];
    const names = new Set();
    history.forEach((entry) => Object.keys(entry.perAccelUtil || {}).forEach((name) => names.add(name)));
    Array.from(names).forEach((name, idx) => {
      perAcceleratorLines.push({
        name,
        type: 'line',
        data: history.map((entry) => {
          const value = entry.perAccelUtil?.[name];
          return Number.isFinite(value) ? value : null;
        }),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: SERIES_COLORS[(idx + 1) % SERIES_COLORS.length] },
      });
    });

    const hostCpuLine = {
      name: 'Host CPU',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.hostCpu) ? entry.hostCpu : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: '#37415C' },
      areaStyle: { color: 'rgba(55, 65, 92, 0.10)' },
    };
    const processCpuLine = {
      name: 'Training process CPU',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.processCpu) ? entry.processCpu : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, type: 'dashed', color: '#0EA5E9' },
    };

    const acceleratorLines = [aggregateLine, ...perAcceleratorLines].filter((series) => hasFiniteValue(series.data));
    const hostProcessLines = [hostCpuLine, processCpuLine].filter((series) => hasFiniteValue(series.data));

    if (preferredLayout === 'accelerator_first' && acceleratorLines.length > 0) {
      return [...acceleratorLines, ...hostProcessLines];
    }
    if (hostProcessLines.length > 0) {
      return [...hostProcessLines, ...acceleratorLines];
    }
    if (acceleratorLines.length > 0) {
      return acceleratorLines;
    }

    const hostRamFallback = {
      name: 'Host RAM',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.ramPressure) ? entry.ramPressure : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: '#B4182D' },
    };
    return hasFiniteValue(hostRamFallback.data) ? [hostRamFallback] : [];
  }, [history, preferredLayout]);

  const memorySeries = useMemo(() => {
    const lines = [];

    const hostRamLine = {
      name: 'System RAM pressure',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.ramPressure) ? entry.ramPressure : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 3, color: '#37415C' },
      areaStyle: { color: 'rgba(55, 65, 92, 0.10)' },
    };
    if (hasFiniteValue(hostRamLine.data)) {
      lines.push(hostRamLine);
    }

    const processRamLine = {
      name: 'Process RAM pressure',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.processRam) ? entry.processRam : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, type: 'dashed', color: '#0EA5E9' },
    };
    if (hasFiniteValue(processRamLine.data)) {
      lines.push(processRamLine);
    }

    const aggregateMemoryLine = {
      name: 'Aggregate memory pressure',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.aggregateMem) ? entry.aggregateMem : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 3, color: '#B4182D' },
      areaStyle: { color: 'rgba(180, 24, 45, 0.12)' },
    };
    if (hasFiniteValue(aggregateMemoryLine.data)) {
      lines.push(aggregateMemoryLine);
    }

    const aggregateFragLine = {
      name: 'Aggregate fragmentation',
      type: 'line',
      data: history.map((entry) => (Number.isFinite(entry.aggregateFrag) ? entry.aggregateFrag : null)),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: '#F97316' },
    };
    if (hasFiniteValue(aggregateFragLine.data)) {
      lines.push(aggregateFragLine);
    }

    const names = new Set();
    history.forEach((entry) => Object.keys(entry.perAccelMem || {}).forEach((name) => names.add(name)));
    Array.from(names).forEach((name, idx) => {
      const series = {
        name: `${name} memory`,
        type: 'line',
        data: history.map((entry) => {
          const value = entry.perAccelMem?.[name];
          return Number.isFinite(value) ? value : null;
        }),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: SERIES_COLORS[(idx + 3) % SERIES_COLORS.length] },
      };
      if (hasFiniteValue(series.data)) {
        lines.push(series);
      }
    });

    lines.push({
      name: 'Pressure threshold (85%)',
      type: 'line',
      data: history.map(() => 85),
      showSymbol: false,
      lineStyle: { width: 1.5, type: 'dotted', color: '#F59E0B' },
      markArea: {
        silent: true,
        itemStyle: { color: 'rgba(245, 158, 11, 0.08)' },
        data: [[{ yAxis: 85 }, { yAxis: 100 }]],
      },
    });

    return lines;
  }, [history]);

  const thermalPowerSeries = useMemo(
    () => [
      {
        name: 'Temperature',
        type: 'line',
        yAxisIndex: 0,
        data: history.map((entry) => (Number.isFinite(entry.aggregateTemp) ? entry.aggregateTemp : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#F97316' },
      },
      {
        name: 'Power draw',
        type: 'line',
        yAxisIndex: 1,
        data: history.map((entry) => (Number.isFinite(entry.aggregatePower) ? entry.aggregatePower : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#22C55E' },
      },
    ],
    [history],
  );

  const throughputSeries = useMemo(
    () => [
      {
        name: 'Disk read MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.diskRead) ? entry.diskRead : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#0EA5E9' },
        areaStyle: { color: 'rgba(14, 165, 233, 0.12)' },
      },
      {
        name: 'Disk write MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.diskWrite) ? entry.diskWrite : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#6366F1' },
      },
      {
        name: 'Network RX MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.netRx) ? entry.netRx : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#10B981' },
      },
      {
        name: 'Network TX MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.netTx) ? entry.netTx : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#F43F5E' },
      },
    ],
    [history],
  );

  const localInsightsSeries = useMemo(
    () => [
      {
        name: 'Samples/sec',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.samplesPerSec) ? entry.samplesPerSec : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#0EA5E9' },
      },
      {
        name: 'Dataloader wait (s)',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.dataloaderWait) ? entry.dataloaderWait : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#6A7286' },
      },
      {
        name: 'Host-to-device transfer (s)',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.hostToDeviceTransfer) ? entry.hostToDeviceTransfer : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.5, type: 'dashed', color: '#F97316' },
      },
      {
        name: 'CPU bottleneck score',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.cpuBottleneck) ? entry.cpuBottleneck : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#B4182D' },
      },
      {
        name: 'Memory pressure',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.memoryPressureScore) ? entry.memoryPressureScore : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#84CC16' },
      },
      {
        name: 'Disk I/O pressure',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.diskPressureScore) ? entry.diskPressureScore : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#14B8A6' },
      },
      {
        name: 'Accelerator starvation',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.acceleratorStarvation) ? entry.acceleratorStarvation : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.5, color: '#F43F5E' },
      },
      {
        name: 'Scaling readiness',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.scalingReadiness) ? entry.scalingReadiness : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#22C55E' },
      },
      {
        name: 'Host CPU',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.hostCpu) ? entry.hostCpu : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.5, color: '#37415C' },
      },
      {
        name: 'Training process CPU',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.processCpu) ? entry.processCpu : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.5, type: 'dashed', color: '#FDA481' },
      },
    ],
    [history],
  );

  const clusterSeries = useMemo(
    () => [
      {
        name: 'Interconnect RX MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.clusterRx) ? entry.clusterRx : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#0EA5E9' },
      },
      {
        name: 'Interconnect TX MB/s',
        type: 'line',
        data: history.map((entry) => (Number.isFinite(entry.clusterTx) ? entry.clusterTx : null)),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#F43F5E' },
      },
    ],
    [history],
  );

  const utilizationOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: utilizationSeries,
        yAxisName: 'Utilization %',
        yAxisMax: 100,
        textColor,
        gridColor,
      }),
    [labels, utilizationSeries, textColor, gridColor],
  );

  const memoryOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: memorySeries,
        yAxisName: 'Pressure %',
        yAxisMax: 100,
        textColor,
        gridColor,
      }),
    [labels, memorySeries, textColor, gridColor],
  );

  const thermalPowerOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: thermalPowerSeries,
        yAxisName: 'Temperature °C',
        yAxisMax: 100,
        textColor,
        gridColor,
        dualAxis: true,
      }),
    [labels, thermalPowerSeries, textColor, gridColor],
  );

  const throughputOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: throughputSeries,
        yAxisName: 'Throughput MB/s',
        textColor,
        gridColor,
      }),
    [labels, throughputSeries, textColor, gridColor],
  );

  const localInsightsOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: localInsightsSeries,
        yAxisName: 'Local metrics',
        textColor,
        gridColor,
      }),
    [labels, localInsightsSeries, textColor, gridColor],
  );

  const clusterOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: clusterSeries,
        yAxisName: 'Interconnect MB/s',
        textColor,
        gridColor,
      }),
    [labels, clusterSeries, textColor, gridColor],
  );

  const acceleratorHistoryByName = useMemo(() => {
    const map = {};
    const names = new Set();
    history.forEach((entry) => Object.keys(entry.perAccelUtil || {}).forEach((name) => names.add(name)));
    Array.from(names).forEach((name) => {
      map[name] = {
        labels,
        values: history.map((entry) => {
          const value = entry.perAccelUtil?.[name];
          return Number.isFinite(value) ? value : null;
        }),
      };
    });
    return map;
  }, [history, labels]);

  if (!runId) {
    return <div className="p-6 text-theme-text-secondary">Run ID is missing.</div>;
  }

  if (isLoading) {
    return <div className="p-6 text-theme-text-secondary">Loading infrastructure telemetry...</div>;
  }

  const panelMode = v2?.panel_mode || (telemetry?.mode === 'distributed' ? 'cluster' : 'local_insights');
  const liveGuard = telemetry?.live_guard;
  const liveGuardReasons = Array.isArray(liveGuard?.reasons) ? liveGuard.reasons : [];
  const systemCpuMetric = getMetric(externalUsage, 'host_cpu_percent') || getMetric(host, 'system_cpu_percent');
  const processCpuMetric = getMetric(externalUsage, 'process_cpu_percent') || getMetric(processMetrics, 'process_cpu_percent');
  const processRamMetric = getMetric(externalUsage, 'process_ram_percent') || getMetric(processMetrics, 'process_ram_percent');
  const systemRamMetric = getMetric(externalUsage, 'host_ram_percent') || getMetric(host, 'system_ram_percent');
  const hasAcceleratorUtilization =
    history.some((entry) => Number.isFinite(entry.aggregateUtil)) ||
    history.some((entry) => Object.values(entry.perAccelUtil || {}).some((value) => Number.isFinite(value)));
  const utilizationIsAccelerator = preferredLayout === 'accelerator_first' && hasAcceleratorUtilization;
  const hasUtilizationSamples = utilizationSeries.some((series) => hasFiniteValue(series.data));
  const hasMemorySamples = memorySeries.some((series) => hasFiniteValue(series.data));
  const hasThroughputSamples = throughputSeries.some((series) => hasFiniteValue(series.data));
  const hasLocalInsightsSamples = localInsightsSeries.some((series) => hasFiniteValue(series.data));
  const hasClusterSamples = clusterSeries.some((series) => hasFiniteValue(series.data));

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="h2 text-theme-text-primary">Infrastructure Telemetry</h1>
            <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
              Adaptive local telemetry with accelerator-aware rendering and graceful fallback behavior.
            </p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Run <span className="font-mono">{runId}</span> · Last update {formatTimestamp(telemetry?.collected_at)} · Source{' '}
              <span className="font-mono">{telemetry?.telemetry_v2?.source?.hostname || 'local-host'}</span>
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={runState?.status || 'idle'} />
            {runTerminal && (
              <span className="badge text-orange-700 bg-orange-500/10 border-orange-500/30 dark:text-orange-300">
                Terminal snapshot
              </span>
            )}
            <span className={`badge border ${modeStyle(v2?.accelerator_mode)}`}>{modeLabel(v2?.accelerator_mode)}</span>
            <span className="badge text-indigo-700 bg-indigo-500/10 border-indigo-500/30 dark:text-indigo-300">
              {telemetry?.mode === 'distributed' ? 'Distributed verified' : 'Local mode'}
            </span>
          </div>
        </div>
        {runState?.status_reason && (
          <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">Run-state note: {runState.status_reason}</p>
        )}
      </div>

      {fetchError && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-700 dark:text-red-300">
          Telemetry refresh failed: {fetchError}
        </div>
      )}

      {!liveGuard?.ok && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/10 p-3 text-sm text-amber-900 dark:text-amber-200">
          <p className="font-semibold">Live guard is active.</p>
          <p className="mt-1">Distributed cluster claims are suppressed until runtime evidence is fresh and consistent.</p>
          {liveGuardReasons.length > 0 && (
            <p className="mt-1 text-xs">
              Reasons: <span className="font-mono">{liveGuardReasons.join(', ')}</span>
            </p>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricTile title="System CPU" icon={Cpu} metric={systemCpuMetric} />
        <MetricTile title="Training Process CPU" icon={Activity} metric={processCpuMetric} />
        <MetricTile title="Training Process RAM" icon={HardDrive} metric={processRamMetric} />
        <MetricTile title="System RAM" icon={HardDrive} metric={systemRamMetric} />
        <MetricTile title="Accelerator Utilization" icon={Gauge} metric={getMetric(aggregate, 'utilization_percent')} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-6">
        <MetricTile title="Accelerator Memory Pressure" icon={Database} metric={getMetric(aggregate, 'memory_pressure_percent')} />
        <MetricTile
          title="Memory Fragmentation"
          icon={Database}
          metric={getMetric(aggregate, 'memory_fragmentation_percent')}
        />
        <MetricTile title="Aggregate Power" icon={Zap} metric={getMetric(aggregate, 'power_watts')} />
        <MetricTile title="Aggregate Temperature" icon={Thermometer} metric={getMetric(aggregate, 'temperature_c')} />
        <MetricTile
          title="Interconnect RX"
          icon={Network}
          metric={panelMode === 'cluster' ? getMetric(clusterMetrics, 'interconnect_rx_mb_s') : getMetric(aggregate, 'interconnect_rx_mb_s')}
        />
        <MetricTile
          title={panelMode === 'cluster' ? 'Cluster Nodes' : 'Scaling Readiness'}
          icon={panelMode === 'cluster' ? Server : BarChart3}
          metric={
            panelMode === 'cluster'
              ? getMetric(clusterMetrics, 'cluster_nodes')
              : getMetric(localInsights, 'scaling_readiness')
          }
        />
      </div>

      {accelerators.length > 0 && (
        <div>
          <h3 className="h3 text-theme-text-primary mb-3">Per-Accelerator Devices</h3>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {accelerators.map((accelerator, idx) => (
              <AcceleratorCard
                key={accelerator?.id || idx}
                accelerator={accelerator}
                historySeries={acceleratorHistoryByName[accelerator?.name || accelerator?.id] || { labels: [], values: [] }}
                gridColor={gridColor}
              />
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card h-[390px] flex flex-col">
          <h3 className="h3 text-theme-text-primary flex items-center gap-2">
            <Activity className="h-5 w-5 text-theme-accent" />
            {utilizationIsAccelerator ? 'Utilization Trends' : 'External Usage Trends'}
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {utilizationIsAccelerator
              ? 'Accelerator utilization with host/process overlays and smooth real-time updates.'
              : 'Host/process usage fallback is active because accelerator series are sparse or unsupported.'}
          </p>
          <div className="mt-3 flex-1 min-h-0">
            {hasUtilizationSamples ? (
              <ReactECharts option={utilizationOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                Waiting for usage samples...
              </div>
            )}
          </div>
        </div>

        <div className="card h-[390px] flex flex-col">
          <h3 className="h3 text-theme-text-primary flex items-center gap-2">
            <Database className="h-5 w-5 text-theme-accent" />
            Memory Pressure
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Host and process RAM are always plotted; accelerator memory overlays appear when available.
          </p>
          <div className="mt-3 flex-1 min-h-0">
            {hasMemorySamples ? (
              <ReactECharts option={memoryOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                Waiting for memory samples...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="card h-[390px] flex flex-col">
          <h3 className="h3 text-theme-text-primary flex items-center gap-2">
            <Thermometer className="h-5 w-5 text-theme-accent" />
            Thermal + Power
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            Dual-axis thermal and power trends where platform counters are available.
          </p>
          <div className="mt-3 flex-1 min-h-0">
            <ReactECharts option={thermalPowerOptions} style={{ height: '100%', width: '100%' }} />
          </div>
        </div>

        <div className="card h-[390px] flex flex-col">
          <h3 className="h3 text-theme-text-primary flex items-center gap-2">
            <Network className="h-5 w-5 text-theme-accent" />
            Disk + Network Throughput
          </h3>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
            {throughputWarmup
              ? 'Rate counters are warming up. Showing safe zero baseline until the next sample.'
              : 'Host I/O throughput for diagnosing local bottlenecks.'}
          </p>
          <div className="mt-3 flex-1 min-h-0">
            {hasThroughputSamples ? (
              <ReactECharts option={throughputOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                Waiting for throughput samples...
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="card h-[390px] flex flex-col">
        <h3 className="h3 text-theme-text-primary flex items-center gap-2">
          {panelMode === 'cluster' ? <Server className="h-5 w-5 text-theme-accent" /> : <BarChart3 className="h-5 w-5 text-theme-accent" />}
          {panelMode === 'cluster' ? 'Cluster / Interconnect Bandwidth' : 'Local Performance Insights'}
        </h3>
        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
          {panelMode === 'cluster'
            ? 'Distributed mode verified. Showing interconnect telemetry for multi-node or multi-rank activity.'
            : 'No verified cluster connection. Showing local bottleneck and scaling readiness insights.'}
        </p>
        <div className="mt-3 flex-1 min-h-0">
          {panelMode === 'cluster' ? (
            hasClusterSamples ? (
              <ReactECharts option={clusterOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                Waiting for cluster interconnect samples...
              </div>
            )
          ) : hasLocalInsightsSamples ? (
            <ReactECharts option={localInsightsOptions} style={{ height: '100%', width: '100%' }} />
          ) : (
            <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
              Waiting for local insight samples...
            </div>
          )}
        </div>
      </div>

      <details className="card">
        <summary className="cursor-pointer select-none text-sm font-semibold text-theme-text-primary">Diagnostics and Probe Details</summary>
        <div className="mt-3 space-y-3 text-xs text-slate-600 dark:text-slate-300">
          <p>
            Run status: <span className="font-mono">{runState?.status || 'unknown'}</span> · Process alive:{' '}
            <span className="font-mono">{runState?.process_alive == null ? 'unknown' : String(runState.process_alive)}</span> · Heartbeat:{' '}
            <span className="font-mono">{formatTimestamp(runState?.heartbeat_ts)}</span>
          </p>
          <p>
            Server source: <span className="font-mono">{v2?.source?.hostname || 'unknown-host'}</span> · PID{' '}
            <span className="font-mono">{v2?.source?.server_pid || '—'}</span>
          </p>
          <div className="rounded-lg border border-theme-border/30 bg-theme-bg/40 p-3">
            <p className="font-semibold text-theme-text-primary">Legend</p>
            <div className="mt-2 grid grid-cols-1 gap-1 md:grid-cols-2">
              {Object.entries(v2?.metric_status_legend || {}).map(([key, label]) => (
                <p key={key}>
                  <span className={`badge ${metricStatusClass(key)}`}>{label}</span> <span className="ml-1 font-mono">{key}</span>
                </p>
              ))}
            </div>
          </div>
          {Array.isArray(v2?.diagnostics) && v2.diagnostics.length > 0 && (
            <div className="rounded-lg border border-theme-border/30 bg-theme-bg/40 p-3">
              <p className="font-semibold text-theme-text-primary">Diagnostics</p>
              <div className="mt-2 space-y-2">
                {v2.diagnostics.map((diag, idx) => (
                  <div key={`${diag?.scope || 'diag'}-${idx}`} className="rounded border border-theme-border/30 p-2">
                    <p>
                      <span className="font-semibold">{diag?.scope || 'scope'}</span> · <span className="font-mono">{diag?.status || 'unknown'}</span>
                    </p>
                    <p className="mt-1">{diag?.message || 'No message'}</p>
                    <p className="mt-1 text-slate-500 dark:text-slate-400">
                      Source: <span className="font-mono">{diag?.source || 'unknown'}</span> · Probe:{' '}
                      <span className="font-mono">{diag?.probe || 'unknown'}</span>
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </details>
    </div>
  );
}
