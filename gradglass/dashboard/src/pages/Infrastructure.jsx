import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Network,
  Server,
  Thermometer,
} from 'lucide-react';
import { useTheme } from '../components/ThemeProvider';
import { fetchInfrastructureTelemetry } from '../api';
import { StatusBadge } from '../components/ui';

const POLL_INTERVAL_MS = 2000;
const HISTORY_LIMIT = 60;

const SERIES_COLORS = ['#FDA481', '#B4182D', '#54162B', '#37415C', '#2F8F9D', '#6A7286', '#22C55E', '#0EA5E9'];

const THRESHOLDS = {
  cpu_bottleneck_score: { warn: 75, critical: 90 },
  memory_pressure: { warn: 85, critical: 95 },
  disk_io_pressure: { warn: 70, critical: 85 },
  accelerator_starvation_idle: { warn: 50, critical: 75 },
  scaling_readiness: { warn: 60, critical: 40 },
};

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

const HEALTH_STYLES = {
  HEALTHY: 'text-emerald-700 bg-emerald-500/10 border-emerald-500/30 dark:text-emerald-300',
  WARNING: 'text-amber-700 bg-amber-500/10 border-amber-500/30 dark:text-amber-300',
  STALLED: 'text-orange-700 bg-orange-500/10 border-orange-500/30 dark:text-orange-300',
  FAILED: 'text-red-700 bg-red-500/10 border-red-500/30 dark:text-red-300',
};

const SEVERITY_STYLES = {
  info: 'text-sky-700 bg-sky-500/10 border-sky-500/30 dark:text-sky-300',
  warning: 'text-amber-700 bg-amber-500/10 border-amber-500/30 dark:text-amber-300',
  critical: 'text-red-700 bg-red-500/10 border-red-500/30 dark:text-red-300',
};

const SEVERITY_ORDER = { critical: 3, warning: 2, info: 1 };

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

function parseFiniteNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    const numeric = Number(trimmed);
    return Number.isFinite(numeric) ? numeric : null;
  }
  return null;
}

function clampPercent(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Number(value)));
}

function modeLabel(mode) {
  if (mode === 'cuda_active') return 'GPU detected (CUDA)';
  if (mode === 'mps_active') return 'GPU detected (MPS)';
  if (mode === 'heterogeneous_active') return 'GPU detected (multi-accelerator)';
  if (mode === 'cpu_only') return 'CPU-only';
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
  return parseFiniteNumber(metric.value);
}

function hasFiniteValue(values) {
  return Array.isArray(values) && values.some((value) => Number.isFinite(value));
}

function toSeries(values) {
  return (values || []).map((value) => (Number.isFinite(value) ? value : null));
}

function historyLatestValue(history, key) {
  for (let idx = history.length - 1; idx >= 0; idx -= 1) {
    const value = history[idx]?.[key];
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function historyValues(history, key) {
  return history.map((row) => (Number.isFinite(row?.[key]) ? row[key] : null));
}

function deriveStepTime(samplesPerSec) {
  if (!Number.isFinite(samplesPerSec) || samplesPerSec <= 0) return null;
  return 1 / samplesPerSec;
}

function severityFromHighThreshold(value, warn, critical) {
  if (!Number.isFinite(value)) return null;
  if (value >= critical) return 'critical';
  if (value >= warn) return 'warning';
  return null;
}

function severityFromLowThreshold(value, warn, critical) {
  if (!Number.isFinite(value)) return null;
  if (value < critical) return 'critical';
  if (value < warn) return 'warning';
  return null;
}

function computeInsights(localInsights, diagnostics, runState) {
  const insightRules = [
    {
      metricKey: 'cpu_bottleneck_score',
      label: 'CPU bottleneck',
      threshold: THRESHOLDS.cpu_bottleneck_score,
      direction: 'high',
      diagnosis: 'Training is CPU-constrained and may underfeed the accelerator.',
      recommendation: 'Increase dataloader workers and prefetching, and reduce heavy CPU-side transforms.',
    },
    {
      metricKey: 'memory_pressure',
      label: 'Memory pressure',
      threshold: THRESHOLDS.memory_pressure,
      direction: 'high',
      diagnosis: 'Host memory is nearing saturation and can trigger paging or instability.',
      recommendation: 'Lower batch size or worker footprint, and close competing memory-heavy processes.',
    },
    {
      metricKey: 'disk_io_pressure',
      label: 'Disk I/O pressure',
      threshold: THRESHOLDS.disk_io_pressure,
      direction: 'high',
      diagnosis: 'Storage throughput is a likely bottleneck for data delivery.',
      recommendation: 'Use local SSD/cache, shard datasets, and reduce random read amplification.',
    },
    {
      metricKey: 'accelerator_starvation_idle',
      label: 'Accelerator starvation',
      threshold: THRESHOLDS.accelerator_starvation_idle,
      direction: 'high',
      diagnosis: 'Accelerator cycles are idle while waiting for input or host-side work.',
      recommendation: 'Tune input pipeline latency (prefetch, pin memory, async transfer) to keep accelerators busy.',
    },
    {
      metricKey: 'scaling_readiness',
      label: 'Scaling readiness',
      threshold: THRESHOLDS.scaling_readiness,
      direction: 'low',
      diagnosis: 'Current bottlenecks indicate poor scaling efficiency if expanded now.',
      recommendation: 'Resolve top bottlenecks before scaling out to additional devices or nodes.',
    },
  ];

  const insights = [];

  insightRules.forEach((rule) => {
    const metric = getMetric(localInsights, rule.metricKey);
    const value = metricValue(metric);
    if (!Number.isFinite(value)) return;

    const severity =
      rule.direction === 'high'
        ? severityFromHighThreshold(value, rule.threshold.warn, rule.threshold.critical)
        : severityFromLowThreshold(value, rule.threshold.warn, rule.threshold.critical);

    const thresholdValue = rule.direction === 'high' ? rule.threshold.warn : rule.threshold.warn;

    insights.push({
      severity: severity || 'info',
      metricKey: rule.metricKey,
      label: rule.label,
      currentValue: value,
      threshold: thresholdValue,
      diagnosis: rule.diagnosis,
      recommendation: rule.recommendation,
    });
  });

  const warnings = insights
    .filter((item) => item.severity === 'warning' || item.severity === 'critical')
    .map((item) => ({
      severity: item.severity,
      title: item.label,
      message: `${item.label} is ${formatNumber(item.currentValue, 1)}. Threshold is ${item.threshold}.`,
      source: 'heuristic',
    }));

  if (Array.isArray(diagnostics)) {
    diagnostics.forEach((diag) => {
      if (!diag || typeof diag !== 'object') return;
      const status = String(diag.status || '').toLowerCase();
      if (!status || status === 'active') return;
      const severity = status.includes('error') || status.includes('missing') ? 'critical' : 'warning';
      warnings.push({
        severity,
        title: `${diag.scope || 'diagnostic'}: ${status.replaceAll('_', ' ')}`,
        message: diag.message || 'Telemetry diagnostic warning.',
        source: 'diagnostic',
      });
    });
  }

  const health = String(runState?.health_state || '').toUpperCase();
  if (health === 'WARNING' || health === 'FAILED' || health === 'STALLED') {
    warnings.push({
      severity: health === 'FAILED' ? 'critical' : 'warning',
      title: `Run health: ${health}`,
      message: 'Run health state indicates instability or degraded operation.',
      source: 'run_state',
    });
  }

  warnings.sort((a, b) => {
    const severityDiff = (SEVERITY_ORDER[b.severity] || 0) - (SEVERITY_ORDER[a.severity] || 0);
    if (severityDiff !== 0) return severityDiff;
    return a.title.localeCompare(b.title);
  });

  const recommendations = Array.from(
    new Set(
      insights
        .filter((item) => item.severity === 'warning' || item.severity === 'critical')
        .map((item) => item.recommendation),
    ),
  );

  const bottleneckRanking = [
    {
      key: 'cpu_bottleneck_score',
      label: 'CPU bottleneck',
      value: metricValue(getMetric(localInsights, 'cpu_bottleneck_score')),
    },
    {
      key: 'memory_pressure',
      label: 'Memory pressure',
      value: metricValue(getMetric(localInsights, 'memory_pressure')),
    },
    {
      key: 'disk_io_pressure',
      label: 'Disk I/O pressure',
      value: metricValue(getMetric(localInsights, 'disk_io_pressure')),
    },
    {
      key: 'accelerator_starvation_idle',
      label: 'Accelerator starvation',
      value: metricValue(getMetric(localInsights, 'accelerator_starvation_idle')),
    },
    {
      key: 'scaling_gap',
      label: 'Scaling readiness gap',
      value: Number.isFinite(metricValue(getMetric(localInsights, 'scaling_readiness')))
        ? Math.max(0, 100 - Number(metricValue(getMetric(localInsights, 'scaling_readiness'))))
        : null,
    },
  ]
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => (b.value || 0) - (a.value || 0));

  return { insights, warnings, recommendations, bottleneckRanking };
}

function MetricBadge({ metric }) {
  const status = metric?.status || 'error';
  const label = metric?.status_label || 'Unknown';
  return <span className={`badge px-2.5 py-0.5 shadow-sm transition-transform hover:scale-105 ${metricStatusClass(status)}`}>{label}</span>;
}

function buildSparklineOption({ labels, values, lineColor, areaColor, gridColor }) {
  return {
    animationDuration: 180,
    animationDurationUpdate: 150,
    grid: { left: 0, right: 0, top: 2, bottom: 0 },
    xAxis: {
      type: 'category',
      data: labels,
      show: false,
    },
    yAxis: {
      type: 'value',
      show: false,
      min: 0,
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    tooltip: { trigger: 'axis' },
    series: [
      {
        type: 'line',
        data: values,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: lineColor },
        areaStyle: { color: areaColor },
      },
    ],
  };
}

function buildLineOptions({
  labels,
  series,
  yAxisName,
  yAxisMax,
  textColor,
  gridColor,
  dualAxis = false,
  showLegend = true,
  gridTop = 36,
}) {
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
    legend: showLegend ? { textStyle: { color: textColor }, top: 0 } : undefined,
    grid: { left: 48, right: dualAxis ? 54 : 24, bottom: 36, top: gridTop },
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
            max: yAxisMax,
            axisLabel: { color: textColor },
            name: yAxisName || undefined,
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
          name: yAxisName || undefined,
          nameTextStyle: { color: textColor },
          splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
        },
    series: normalizedSeries,
  };
}

function CoreHealthCard({ title, icon: Icon, metric, labels, values, lineColor, areaColor, gridColor }) {
  const value = metricValue(metric);
  const displayValue = metric?.display || (Number.isFinite(value) ? `${formatNumber(value, 1)}%` : '—');
  const progressWidth = `${clampPercent(value)}%`;

  const hasTrend = hasFiniteValue(values);

  const sparklineOption = useMemo(
    () =>
      buildSparklineOption({
        labels,
        values,
        lineColor,
        areaColor,
        gridColor,
      }),
    [labels, values, lineColor, areaColor, gridColor],
  );

  return (
    <div className="glass-card group flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between opacity-90 transition-opacity duration-300 group-hover:opacity-100">
          <p className="text-sm font-semibold uppercase tracking-wide text-theme-text-secondary transition-colors group-hover:text-theme-primary">{title}</p>
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-theme-primary/10 text-theme-primary transition-transform duration-300 group-hover:scale-110">
            <Icon className="h-4 w-4" />
          </div>
        </div>

        <div className="mt-4 flex items-end justify-between gap-3">
          <p className="text-3xl font-extrabold tracking-tight text-theme-text-primary">{displayValue}</p>
          <MetricBadge metric={metric} />
        </div>

        <div className="mt-3 h-2 rounded-full bg-theme-border/40 overflow-hidden">
          <div className="h-full rounded-full" style={{ width: progressWidth, backgroundColor: lineColor }} />
        </div>

        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">Updated {formatTimestamp(metric?.timestamp)}</p>
      </div>

      <div className="mt-4 h-14">
        {hasTrend ? (
          <ReactECharts option={sparklineOption} style={{ width: '100%', height: '100%' }} />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-slate-500 dark:text-slate-400">
            Waiting for samples...
          </div>
        )}
      </div>
    </div>
  );
}

function SeverityBadge({ severity }) {
  const normalized = severity || 'info';
  return (
    <span className={`badge px-2.5 py-0.5 font-bold tracking-wider shadow-sm transition-transform hover:scale-105 ${SEVERITY_STYLES[normalized] || SEVERITY_STYLES.info}`}>
      {normalized.toUpperCase()}
    </span>
  );
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
        const external = v2?.external_usage || {};
        const local = v2?.local_performance_insights || {};
        const graphHints = v2?.graph_hints || {};

        const ts = Number(payload?.collected_at) || Date.now() / 1000;

        const perAccelUtil = {};
        const perAccelMem = {};
        const accelerators = Array.isArray(v2?.accelerators) ? v2.accelerators : [];

        accelerators.forEach((acc, idx) => {
          const name = acc?.name || acc?.id || `Accelerator ${idx + 1}`;
          const utilValue = metricValue(getMetric(acc?.metrics, 'utilization_percent'));
          const memValue = metricValue(getMetric(acc?.metrics, 'memory_pressure_percent'));
          if (utilValue != null) perAccelUtil[name] = utilValue;
          if (memValue != null) perAccelMem[name] = memValue;
        });

        const samplesPerSec = metricValue(getMetric(local, 'samples_per_sec'));

        const point = {
          ts,
          hostCpu:
            metricValue(getMetric(external, 'host_cpu_percent')) ??
            metricValue(getMetric(host, 'system_cpu_percent')),
          processCpu:
            metricValue(getMetric(external, 'process_cpu_percent')) ??
            metricValue(getMetric(process, 'process_cpu_percent')),
          processRam:
            metricValue(getMetric(external, 'process_ram_percent')) ??
            metricValue(getMetric(process, 'process_ram_percent')),
          systemRam:
            metricValue(getMetric(external, 'host_ram_percent')) ??
            metricValue(getMetric(host, 'system_ram_percent')),
          samplesPerSec,
          stepTime: deriveStepTime(samplesPerSec),
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
          scalingReadiness: metricValue(getMetric(local, 'scaling_readiness')),
          cpuBottleneck: metricValue(getMetric(local, 'cpu_bottleneck_score')),
          memoryPressure: metricValue(getMetric(local, 'memory_pressure')),
          diskPressure: metricValue(getMetric(local, 'disk_io_pressure')),
          acceleratorStarvation: metricValue(getMetric(local, 'accelerator_starvation_idle')),
          aggregateUtil: metricValue(getMetric(aggregate, 'utilization_percent')),
          aggregateMem: metricValue(getMetric(aggregate, 'memory_pressure_percent')),
          aggregatePower: metricValue(getMetric(aggregate, 'power_watts')),
          aggregateTemp: metricValue(getMetric(aggregate, 'temperature_c')),
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
            ['processCpu', 'processRam', 'samplesPerSec', 'stepTime'].forEach((key) => {
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
  const host = v2?.host_metrics || {};
  const processMetrics = v2?.training_process_metrics || {};
  const externalUsage = v2?.external_usage || {};
  const localInsights = v2?.local_performance_insights || {};
  const diagnostics = Array.isArray(v2?.diagnostics) ? v2.diagnostics : [];
  const runState = v2?.run_state || {};
  const graphHints = v2?.graph_hints || {};
  const accelerators = Array.isArray(v2?.accelerators) ? v2.accelerators : [];
  const runTerminal = Boolean(graphHints?.run_terminal);
  const throughputWarmup = Boolean(graphHints?.throughput_warmup);

  const labels = history.map((entry) => formatTimestamp(entry.ts));

  const systemCpuMetric = getMetric(externalUsage, 'host_cpu_percent') || getMetric(host, 'system_cpu_percent');
  const processCpuMetric = getMetric(externalUsage, 'process_cpu_percent') || getMetric(processMetrics, 'process_cpu_percent');
  const processRamMetric = getMetric(externalUsage, 'process_ram_percent') || getMetric(processMetrics, 'process_ram_percent');
  const systemRamMetric = getMetric(externalUsage, 'host_ram_percent') || getMetric(host, 'system_ram_percent');

  const utilizationSeries = useMemo(() => {
    const aggregateLine = {
      name: 'Aggregate utilization',
      type: 'line',
      data: toSeries(historyValues(history, 'aggregateUtil')),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 3, color: '#FDA481' },
      areaStyle: { color: 'rgba(253, 164, 129, 0.12)' },
    };

    const hostCpuLine = {
      name: 'Host CPU',
      type: 'line',
      data: toSeries(historyValues(history, 'hostCpu')),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, color: '#37415C' },
    };

    const processCpuLine = {
      name: 'Training process CPU',
      type: 'line',
      data: toSeries(historyValues(history, 'processCpu')),
      smooth: true,
      showSymbol: false,
      lineStyle: { width: 2, type: 'dashed', color: '#0EA5E9' },
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

    return [aggregateLine, hostCpuLine, processCpuLine, ...perAcceleratorLines].filter((series) => hasFiniteValue(series.data));
  }, [history]);

  const samplesStepSeries = useMemo(
    () => [
      {
        name: 'Samples/sec',
        type: 'line',
        yAxisIndex: 0,
        data: toSeries(historyValues(history, 'samplesPerSec')),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#0EA5E9' },
        areaStyle: { color: 'rgba(14, 165, 233, 0.14)' },
      },
      {
        name: 'Step time (s)',
        type: 'line',
        yAxisIndex: 1,
        data: toSeries(historyValues(history, 'stepTime')),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, type: 'dashed', color: '#F97316' },
      },
    ],
    [history],
  );

  const throughputSeries = useMemo(
    () => [
      {
        name: 'Disk read MB/s',
        type: 'bar',
        stack: 'disk',
        data: toSeries(historyValues(history, 'diskRead')),
        itemStyle: { color: '#0EA5E9' },
      },
      {
        name: 'Disk write MB/s',
        type: 'bar',
        stack: 'disk',
        data: toSeries(historyValues(history, 'diskWrite')),
        itemStyle: { color: '#6366F1' },
      },
      {
        name: 'Network RX MB/s',
        type: 'bar',
        stack: 'network',
        data: toSeries(historyValues(history, 'netRx')),
        itemStyle: { color: '#10B981' },
      },
      {
        name: 'Network TX MB/s',
        type: 'bar',
        stack: 'network',
        data: toSeries(historyValues(history, 'netTx')),
        itemStyle: { color: '#F43F5E' },
      },
    ],
    [history],
  );

  const tempPowerSeries = useMemo(
    () => [
      {
        name: 'Temperature',
        type: 'line',
        yAxisIndex: 0,
        data: toSeries(historyValues(history, 'aggregateTemp')),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#F97316' },
      },
      {
        name: 'Power draw',
        type: 'line',
        yAxisIndex: 1,
        data: toSeries(historyValues(history, 'aggregatePower')),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#22C55E' },
      },
    ],
    [history],
  );

  const scalingTrendSeries = useMemo(
    () => [
      {
        name: 'Scaling readiness',
        type: 'line',
        data: toSeries(historyValues(history, 'scalingReadiness')),
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2, color: '#22C55E' },
        areaStyle: { color: 'rgba(34, 197, 94, 0.14)' },
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

  const samplesStepOptions = useMemo(
    () => ({
      animationDuration: 260,
      animationDurationUpdate: 220,
      tooltip: { trigger: 'axis' },
      legend: { textStyle: { color: textColor }, top: 0 },
      grid: { left: 52, right: 56, bottom: 36, top: 36 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: textColor },
        axisLine: { lineStyle: { color: gridColor } },
      },
      yAxis: [
        {
          type: 'value',
          min: 0,
          axisLabel: { color: textColor },
          name: 'Samples/s',
          nameTextStyle: { color: textColor },
          splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
        },
        {
          type: 'value',
          min: 0,
          axisLabel: { color: textColor },
          name: 'Step time (s)',
          nameTextStyle: { color: textColor },
          splitLine: { show: false },
        },
      ],
      series: samplesStepSeries,
    }),
    [labels, samplesStepSeries, textColor, gridColor],
  );

  const throughputOptions = useMemo(
    () => ({
      animationDuration: 260,
      animationDurationUpdate: 220,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { textStyle: { color: textColor }, top: 0 },
      grid: { left: 48, right: 24, bottom: 36, top: 36 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: textColor },
        axisLine: { lineStyle: { color: gridColor } },
      },
      yAxis: {
        type: 'value',
        min: 0,
        axisLabel: { color: textColor },
        name: 'MB/s',
        nameTextStyle: { color: textColor },
        splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      },
      series: throughputSeries,
    }),
    [labels, throughputSeries, textColor, gridColor],
  );

  const scalingTrendOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: scalingTrendSeries,
        yAxisName: undefined,
        yAxisMax: 100,
        textColor,
        gridColor,
        showLegend: false,
        gridTop: 52,
      }),
    [labels, scalingTrendSeries, textColor, gridColor],
  );

  const tempPowerOptions = useMemo(
    () =>
      buildLineOptions({
        labels,
        series: tempPowerSeries,
        yAxisName: 'Temperature °C',
        yAxisMax: 100,
        textColor,
        gridColor,
        dualAxis: true,
      }),
    [labels, tempPowerSeries, textColor, gridColor],
  );

  const memoryByAccelerator = useMemo(() => {
    const rows = accelerators
      .map((accelerator, idx) => {
        const name = accelerator?.name || accelerator?.id || `Accelerator ${idx + 1}`;
        const value = metricValue(getMetric(accelerator?.metrics, 'memory_pressure_percent'));
        return {
          name,
          value: Number.isFinite(value) ? value : null,
        };
      })
      .filter((row) => Number.isFinite(row.value));
    return rows;
  }, [accelerators]);

  const vramPressureOptions = useMemo(
    () => ({
      animationDuration: 240,
      animationDurationUpdate: 220,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 48, right: 20, top: 18, bottom: 36 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: textColor, formatter: '{value}%' },
        splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      },
      yAxis: {
        type: 'category',
        data: memoryByAccelerator.map((row) => row.name),
        axisLabel: { color: textColor },
      },
      series: [
        {
          type: 'bar',
          data: memoryByAccelerator.map((row, idx) => ({
            value: row.value,
            itemStyle: { color: SERIES_COLORS[idx % SERIES_COLORS.length] },
          })),
          label: {
            show: true,
            position: 'right',
            formatter: ({ value }) => (Number.isFinite(value) ? `${Number(value).toFixed(1)}%` : '—'),
            color: textColor,
          },
        },
      ],
    }),
    [memoryByAccelerator, textColor, gridColor],
  );

  const currentScalingReadiness = metricValue(getMetric(localInsights, 'scaling_readiness'));

  const scalingGaugeOptions = useMemo(
    () => ({
      series: [
        {
          type: 'gauge',
          min: 0,
          max: 100,
          splitNumber: 5,
          progress: {
            show: true,
            width: 14,
            itemStyle: {
              color:
                currentScalingReadiness == null
                  ? '#6A7286'
                  : currentScalingReadiness < THRESHOLDS.scaling_readiness.critical
                    ? '#F43F5E'
                    : currentScalingReadiness < THRESHOLDS.scaling_readiness.warn
                      ? '#F59E0B'
                      : '#22C55E',
            },
          },
          axisLine: {
            lineStyle: {
              width: 14,
              color: [[1, theme === 'dark' ? 'rgba(255,255,255,0.15)' : 'rgba(55,65,92,0.15)']],
            },
          },
          pointer: { show: false },
          axisTick: { distance: -18, splitNumber: 4, lineStyle: { color: gridColor, width: 1 } },
          splitLine: { distance: -20, length: 8, lineStyle: { color: gridColor, width: 1.5 } },
          axisLabel: { color: textColor, distance: 14 },
          detail: {
            valueAnimation: true,
            formatter: (value) => `${Number(value).toFixed(1)}%`,
            color: textColor,
            fontSize: 20,
            offsetCenter: [0, '20%'],
          },
          title: { show: false, color: textColor, offsetCenter: [0, '62%'], fontSize: 12 },
          data: [{ value: currentScalingReadiness || 0, name: 'Scaling readiness' }],
        },
      ],
    }),
    [currentScalingReadiness, theme, gridColor, textColor],
  );

  const insightsModel = useMemo(
    () => computeInsights(localInsights, diagnostics, runState),
    [localInsights, diagnostics, runState],
  );

  const bottleneckOptions = useMemo(
    () => ({
      animationDuration: 240,
      animationDurationUpdate: 220,
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 142, right: 16, top: 16, bottom: 24 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: { color: textColor, formatter: '{value}%' },
        splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      },
      yAxis: {
        type: 'category',
        data: insightsModel.bottleneckRanking.map((item) => item.label),
        axisLabel: { color: textColor },
      },
      series: [
        {
          type: 'bar',
          data: insightsModel.bottleneckRanking.map((item) => item.value),
          itemStyle: {
            color: ({ value }) => {
              if (value >= 85) return '#F43F5E';
              if (value >= 60) return '#F59E0B';
              return '#22C55E';
            },
          },
          label: {
            show: true,
            position: 'right',
            formatter: ({ value }) => (Number.isFinite(value) ? `${Number(value).toFixed(1)}%` : '—'),
            color: textColor,
          },
        },
      ],
    }),
    [insightsModel.bottleneckRanking, textColor, gridColor],
  );

  const hasUtilizationSamples = utilizationSeries.some((series) => hasFiniteValue(series.data));
  const hasSamplesTrend = samplesStepSeries.some((series) => hasFiniteValue(series.data));
  const hasThroughputSamples = throughputSeries.some((series) => hasFiniteValue(series.data));
  const hasScalingTrend = scalingTrendSeries.some((series) => hasFiniteValue(series.data));
  const hasTempPowerSamples = tempPowerSeries.some((series) => hasFiniteValue(series.data));
  const samplesPerSecMetric = getMetric(localInsights, 'samples_per_sec');
  const samplesTrendMessage = throughputWarmup
    ? 'Samples/sec counters are warming up. Initial values may lag.'
    : samplesPerSecMetric?.status && samplesPerSecMetric.status !== 'active'
      ? `Samples/sec unavailable: ${samplesPerSecMetric.status_label || 'Not detected'}.`
      : 'Waiting for samples-per-second data...';

  const activeAcceleratorMode = ['cuda_active', 'mps_active', 'heterogeneous_active'].includes(v2?.accelerator_mode);
  const hasAcceleratorSamples = history.some((entry) => Number.isFinite(entry.aggregateUtil)) || memoryByAccelerator.length > 0;
  const showConditionalHardware = activeAcceleratorMode || hasAcceleratorSamples;

  if (!runId) {
    return <div className="p-6 text-theme-text-secondary">Run ID is missing.</div>;
  }

  if (isLoading) {
    return <div className="p-6 text-theme-text-secondary">Loading infrastructure telemetry...</div>;
  }

  const latestSamplesPerSec = historyLatestValue(history, 'samplesPerSec');
  const latestStepTime = historyLatestValue(history, 'stepTime');

  return (
    <div className="space-y-8">
      <div className="glass-panel">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-theme-text-primary to-theme-primary">Infrastructure Telemetry</h1>
            <p className="mt-4 max-w-3xl text-base leading-relaxed text-theme-text-secondary">
              Real-time monitoring of hardware utilization, memory pressure, and cluster scaling efficiency.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-2 text-xs font-medium text-theme-text-muted">
              <span>Run <span className="font-mono bg-theme-bg/50 px-1 py-0.5 rounded">{runId}</span></span>
              <span>·</span>
              <span>Last update <span className="font-mono">{formatTimestamp(telemetry?.collected_at)}</span></span>
              <span>·</span>
              <span>Source <span className="font-mono bg-theme-bg/50 px-1 py-0.5 rounded">{v2?.source?.hostname || 'local-host'}</span></span>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="badge text-indigo-700 bg-indigo-500/10 border-indigo-500/30 dark:text-indigo-300">
              Mode: {telemetry?.mode === 'distributed' ? 'Distributed' : 'Local'}
            </span>
            <span className={`badge border ${modeStyle(v2?.accelerator_mode)}`}>{modeLabel(v2?.accelerator_mode)}</span>
            <StatusBadge status={runState?.status || 'idle'} />
            <span className={`badge border ${HEALTH_STYLES[String(runState?.health_state || 'WARNING').toUpperCase()] || HEALTH_STYLES.WARNING}`}>
              Health: {String(runState?.health_state || 'UNKNOWN').toUpperCase()}
            </span>
            {runTerminal && (
              <span className="badge text-orange-700 bg-orange-500/10 border-orange-500/30 dark:text-orange-300">
                Terminal snapshot
              </span>
            )}
          </div>
        </div>

        {runState?.status_reason && (
          <p className="mt-4 text-sm font-medium text-theme-text-secondary border-l-4 border-theme-primary pl-3 bg-theme-primary/5 py-2">
            Run-state note: {runState.status_reason}
          </p>
        )}
      </div>

      {fetchError && (
        <div className="rounded-2xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-700 dark:text-red-300">
          Telemetry refresh failed: {fetchError}
        </div>
      )}

      <div className="glass-panel">
        <div className="mb-6 flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Cpu className="h-6 w-6 text-theme-accent" />
              <h2 className="text-2xl font-bold text-theme-text-primary">Core Health</h2>
            </div>
          </div>
          <div className="rounded-2xl border border-theme-primary/30 bg-theme-primary/5 p-5">
            <h3 className="text-sm font-bold text-theme-primary mb-2">🎓 What to look for: Core Resource Saturation</h3>
            <p className="text-sm text-theme-text-secondary">If Training Process CPU hits 100%, your dataloaders are likely blocking the accelerator. High System RAM usage indicates memory leaks or excessively large prefetch buffers, which will crash your run. Keep these below 80% to ensure smooth scaling.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <CoreHealthCard
            title="Training Process CPU"
            icon={Activity}
            metric={processCpuMetric}
            labels={labels}
            values={toSeries(historyValues(history, 'processCpu'))}
            lineColor="#FDA481"
            areaColor="rgba(253, 164, 129, 0.18)"
            gridColor={gridColor}
          />

          <CoreHealthCard
            title="Training Process RAM"
            icon={HardDrive}
            metric={processRamMetric}
            labels={labels}
            values={toSeries(historyValues(history, 'processRam'))}
            lineColor="#B4182D"
            areaColor="rgba(180, 24, 45, 0.16)"
            gridColor={gridColor}
          />

          <CoreHealthCard
            title="System CPU"
            icon={Cpu}
            metric={systemCpuMetric}
            labels={labels}
            values={toSeries(historyValues(history, 'hostCpu'))}
            lineColor="#37415C"
            areaColor="rgba(55, 65, 92, 0.15)"
            gridColor={gridColor}
          />

          <CoreHealthCard
            title="System RAM"
            icon={Database}
            metric={systemRamMetric}
            labels={labels}
            values={toSeries(historyValues(history, 'systemRam'))}
            lineColor="#0EA5E9"
            areaColor="rgba(14, 165, 233, 0.16)"
            gridColor={gridColor}
          />
        </div>
      </div>

      <div className="glass-panel">
        <div className="mb-6">
          <div className="flex items-center gap-3">
            <Network className="h-6 w-6 text-theme-accent" />
            <h2 className="text-2xl font-bold text-theme-text-primary">Performance</h2>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="glass-card h-[390px] flex flex-col">
            <h3 className="h3 text-theme-text-primary flex items-center gap-2">
              <Activity className="h-5 w-5 text-theme-accent" />
              Step Cadence
            </h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Samples/sec and derived step time (`step_time = 1 / samples_per_sec`) updated every 2s.
            </p>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="badge text-sky-700 bg-sky-500/10 border-sky-500/30 dark:text-sky-300">
                Samples/s: {Number.isFinite(latestSamplesPerSec) ? formatNumber(latestSamplesPerSec, 2) : '—'}
              </span>
              <span className="badge text-orange-700 bg-orange-500/10 border-orange-500/30 dark:text-orange-300">
                Step time: {Number.isFinite(latestStepTime) ? `${formatNumber(latestStepTime, 4)}s` : '—'}
              </span>
            </div>
            <div className="mt-3 flex-1 min-h-0">
              {hasSamplesTrend ? (
                <ReactECharts option={samplesStepOptions} style={{ height: '100%', width: '100%' }} />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                  {samplesTrendMessage}
                </div>
              )}
            </div>
          </div>

          <div className="glass-card h-[390px] flex flex-col">
            <h3 className="h3 text-theme-text-primary flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-theme-accent" />
              Disk + Network Throughput Mix
            </h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Grouped stacked bars for storage and network throughput.
            </p>
            {throughputWarmup && (
              <p className="mt-2 text-xs text-amber-700 dark:text-amber-300">
                Throughput counters are warming up. Early samples may include safe zero baselines.
              </p>
            )}
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
      </div>

      <div className="glass-panel">
        <div className="mb-6">
          <div className="flex items-center gap-3">
            <Gauge className="h-6 w-6 text-theme-accent" />
            <h2 className="text-2xl font-bold text-theme-text-primary">Scaling Readiness</h2>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="glass-card h-[390px] flex flex-col">
            <h3 className="h3 text-theme-text-primary">Readiness Gauge + Trend</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Warning below 60%, critical below 40%.
            </p>
            <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2 flex-1 min-h-0">
              <div className="h-full min-h-[160px]">
                <ReactECharts option={scalingGaugeOptions} style={{ width: '100%', height: '100%' }} />
                <p className="mt-1 text-center text-xs text-slate-500 dark:text-slate-400">Scaling readiness</p>
              </div>
              <div className="h-full min-h-[160px]">
                {hasScalingTrend ? (
                  <ReactECharts option={scalingTrendOptions} style={{ width: '100%', height: '100%' }} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                    Waiting for scaling trend samples...
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="glass-card h-[390px] flex flex-col">
            <h3 className="h3 text-theme-text-primary">Bottleneck Score Snapshot</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Heuristic penalty scores feeding scaling readiness.
            </p>

            <div className="mt-4 space-y-4">
              {[
                { label: 'CPU bottleneck', key: 'cpuBottleneck', warn: THRESHOLDS.cpu_bottleneck_score.warn, critical: THRESHOLDS.cpu_bottleneck_score.critical },
                { label: 'Memory pressure', key: 'memoryPressure', warn: THRESHOLDS.memory_pressure.warn, critical: THRESHOLDS.memory_pressure.critical },
                { label: 'Disk I/O pressure', key: 'diskPressure', warn: THRESHOLDS.disk_io_pressure.warn, critical: THRESHOLDS.disk_io_pressure.critical },
                {
                  label: 'Accelerator starvation',
                  key: 'acceleratorStarvation',
                  warn: THRESHOLDS.accelerator_starvation_idle.warn,
                  critical: THRESHOLDS.accelerator_starvation_idle.critical,
                },
              ].map((row) => {
                const value = historyLatestValue(history, row.key);
                const severity = severityFromHighThreshold(value, row.warn, row.critical);
                const barColor = severity === 'critical' ? '#F43F5E' : severity === 'warning' ? '#F59E0B' : '#22C55E';

                return (
                  <div key={row.key}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span className="text-theme-text-primary">{row.label}</span>
                      <span className="font-mono text-theme-text-primary">{Number.isFinite(value) ? `${formatNumber(value, 1)}%` : '—'}</span>
                    </div>
                    <div className="h-2 rounded-full bg-theme-border/40 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{ width: `${clampPercent(value)}%`, backgroundColor: barColor }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel">
        <div className="mb-6">
          <div className="flex items-center gap-3">
            <Server className="h-6 w-6 text-theme-accent" />
            <h2 className="text-2xl font-bold text-theme-text-primary">Conditional Hardware</h2>
          </div>
        </div>

        {showConditionalHardware ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="glass-card h-[360px] flex flex-col lg:col-span-2">
              <h3 className="h3 text-theme-text-primary">GPU/MPS Utilization</h3>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Aggregate and per-accelerator utilization trends.
              </p>
              <div className="mt-3 flex-1 min-h-0">
                {hasUtilizationSamples ? (
                  <ReactECharts option={utilizationOptions} style={{ width: '100%', height: '100%' }} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                    Waiting for accelerator utilization samples...
                  </div>
                )}
              </div>
            </div>

            <div className="glass-card h-[360px] flex flex-col">
              <h3 className="h3 text-theme-text-primary">VRAM Pressure</h3>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Current memory pressure by accelerator.
              </p>
              <div className="mt-3 flex-1 min-h-0">
                {memoryByAccelerator.length > 0 ? (
                  <ReactECharts option={vramPressureOptions} style={{ width: '100%', height: '100%' }} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                    VRAM pressure is unavailable on this platform.
                  </div>
                )}
              </div>
            </div>

            <div className="glass-card h-[360px] flex flex-col lg:col-span-3">
              <h3 className="h3 text-theme-text-primary flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-theme-accent" />
                Temperature + Power
              </h3>
              <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                Dual-axis thermal and power trends from aggregate accelerator telemetry.
              </p>
              <div className="mt-3 flex-1 min-h-0">
                {hasTempPowerSamples ? (
                  <ReactECharts option={tempPowerOptions} style={{ width: '100%', height: '100%' }} />
                ) : (
                  <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                    Thermal/power counters are unavailable on this platform.
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="glass-card">
            <div className="flex items-center gap-2 text-slate-700 dark:text-slate-300">
              <Server className="h-4 w-4 text-theme-accent" />
              <p className="text-sm font-medium">No active accelerator telemetry detected.</p>
            </div>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">
              Running in CPU-only mode or accelerator counters are unsupported. Core health and performance rows remain active.
            </p>
          </div>
        )}
      </div>

      <div className="glass-panel">
        <div className="mb-6 flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <AlertTriangle className="h-6 w-6 text-theme-accent" />
            <h2 className="text-2xl font-bold text-theme-text-primary">Insights</h2>
          </div>
          <div className="rounded-2xl border border-theme-primary/30 bg-theme-primary/5 p-5">
            <h3 className="text-sm font-bold text-theme-primary mb-2">🎓 What to look for: Diagnostics & Bottlenecks</h3>
            <p className="text-sm text-theme-text-secondary">Look at the Bottleneck Diagnosis chart. High scores mean you are wasting money on hardware. If you see 'Disk I/O pressure', consider caching datasets in RAM. If you see 'Accelerator starvation', it means your GPUs are waiting on CPUs to feed them batches—optimize your dataloaders immediately.</p>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-3">
          <div className="glass-card h-[360px] flex flex-col">
            <h3 className="h3 text-theme-text-primary">Bottleneck Diagnosis</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Higher bars indicate stronger contribution to training slowdown.
            </p>
            <div className="mt-3 flex-1 min-h-0">
              {insightsModel.bottleneckRanking.length > 0 ? (
                <ReactECharts option={bottleneckOptions} style={{ width: '100%', height: '100%' }} />
              ) : (
                <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                  Bottleneck metrics are not available yet.
                </div>
              )}
            </div>
          </div>

          <div className="glass-card h-[360px] flex flex-col">
            <h3 className="h3 text-theme-text-primary">Warnings</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Heuristic and diagnostic warnings derived from current telemetry.
            </p>
            <div className="mt-3 flex-1 overflow-auto space-y-2">
              {insightsModel.warnings.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-emerald-700 dark:text-emerald-300">
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  No active warnings.
                </div>
              ) : (
                insightsModel.warnings.map((warning, idx) => (
                  <div key={`${warning.title}-${idx}`} className="rounded-lg border border-theme-border/40 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium text-theme-text-primary">{warning.title}</p>
                      <SeverityBadge severity={warning.severity} />
                    </div>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{warning.message}</p>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                      Source: <span className="font-mono">{warning.source}</span>
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="glass-card h-[360px] flex flex-col">
            <h3 className="h3 text-theme-text-primary">Recommended Fixes</h3>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              Actionable next steps based on active warning rules.
            </p>
            <div className="mt-3 flex-1 overflow-auto">
              {insightsModel.recommendations.length === 0 ? (
                <div className="flex h-full items-center justify-center text-sm text-slate-500 dark:text-slate-400">
                  No intervention recommended right now.
                </div>
              ) : (
                <ol className="space-y-2 text-sm text-theme-text-primary list-decimal pl-5">
                  {insightsModel.recommendations.map((item, idx) => (
                    <li key={`fix-${idx}`}>{item}</li>
                  ))}
                </ol>
              )}
            </div>
          </div>
        </div>
      </div>

      <details className="glass-panel">
        <summary className="cursor-pointer select-none text-xl font-bold text-theme-text-primary outline-none">Diagnostics and Probe Details</summary>
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
            <p className="font-semibold text-theme-text-primary">Metric Status Legend</p>
            <div className="mt-2 grid grid-cols-1 gap-1 md:grid-cols-2">
              {Object.entries(v2?.metric_status_legend || {}).map(([key, label]) => (
                <p key={key}>
                  <span className={`badge ${metricStatusClass(key)}`}>{label}</span> <span className="ml-1 font-mono">{key}</span>
                </p>
              ))}
            </div>
          </div>

          {diagnostics.length > 0 && (
            <div className="rounded-lg border border-theme-border/30 bg-theme-bg/40 p-3">
              <p className="font-semibold text-theme-text-primary">Diagnostics</p>
              <div className="mt-2 space-y-2">
                {diagnostics.map((diag, idx) => (
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
