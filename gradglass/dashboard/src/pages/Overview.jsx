import React, { useEffect } from 'react';
import { useParams } from 'react-router-dom';
import useRunStore from '../store/useRunStore';
import { useTheme } from '../components/ThemeProvider';
import { StatusBadge } from '../components/ui';
import ReactECharts from 'echarts-for-react';
import { ShieldCheck, ShieldAlert, AlertTriangle, Zap, Target, Clock, TrendingDown } from 'lucide-react';
import {
  DEFAULT_METRIC_EXCLUDE_KEYS,
  extractNumericSeries,
  formatMetricKeyLabel,
  latestSeriesValue,
  resolvePrimaryMetricKey,
} from '../utils';

const HEALTH_STYLES = {
  HEALTHY: {
    icon: ShieldCheck,
    text: 'text-emerald-600 dark:text-emerald-500',
    border: 'border-emerald-300 dark:border-emerald-500/30',
  },
  WARNING: {
    icon: ShieldAlert,
    text: 'text-amber-600 dark:text-amber-500',
    border: 'border-amber-300 dark:border-amber-500/30',
  },
  STALLED: {
    icon: AlertTriangle,
    text: 'text-orange-600 dark:text-orange-400',
    border: 'border-orange-300 dark:border-orange-500/30',
  },
  FAILED: {
    icon: AlertTriangle,
    text: 'text-red-600 dark:text-red-500',
    border: 'border-red-300 dark:border-red-500/30',
  },
};

const TRAIN_FALLBACK_PRIORITY = [
  'train_accuracy',
  'accuracy',
  'acc',
  'train_acc',
  'train_f1',
  'f1',
  'train_auc',
  'auc',
  'train_r2',
  'r2',
  'train_score',
  'score',
];

const VAL_FALLBACK_PRIORITY = [
  'val_accuracy',
  'validation_accuracy',
  'test_accuracy',
  'val_acc',
  'validation_acc',
  'test_acc',
  'val_f1',
  'validation_f1',
  'test_f1',
  'f1',
  'val_auc',
  'validation_auc',
  'test_auc',
  'auc',
  'val_r2',
  'validation_r2',
  'test_r2',
  'r2',
  'val_score',
  'validation_score',
  'test_score',
  'score',
];

function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return '-';
  const s = Math.max(0, Math.floor(seconds));
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = s % 60;

  if (hrs > 0) return `${hrs}h ${mins}m ${secs}s`;
  if (mins > 0) return `${mins}m ${secs}s`;
  return `${secs}s`;
}

function formatFloat(value, digits = 4) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

export default function Overview() {
  const { runId } = useParams();
  const { setActiveRun, metadata, metrics, alerts, overview } = useRunStore();
  const { theme } = useTheme();

  useEffect(() => {
    if (runId) {
      setActiveRun(runId);
    }
  }, [runId, setActiveRun]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.05)';

  const trainLoss = overview?.loss_history?.length ? overview.loss_history : extractNumericSeries(metrics, 'loss');
  const valLoss = overview?.val_loss_history?.length ? overview.val_loss_history : extractNumericSeries(metrics, 'val_loss');
  const lrData = overview?.lr_history?.length ? overview.lr_history : extractNumericSeries(metrics, 'lr');

  const hasLossSeries = trainLoss.length > 0 || valLoss.length > 0;

  const trainFallbackKey = !hasLossSeries
    ? resolvePrimaryMetricKey(metrics, {
        priorityKeys: TRAIN_FALLBACK_PRIORITY,
        includeTokens: ['train', 'accuracy', 'acc', 'f1', 'auc', 'r2', 'score', 'precision', 'recall'],
        excludeTokens: ['val', 'validation', 'test'],
        excludeKeys: DEFAULT_METRIC_EXCLUDE_KEYS,
      })
    : null;

  let valFallbackKey = !hasLossSeries
    ? resolvePrimaryMetricKey(metrics, {
        priorityKeys: VAL_FALLBACK_PRIORITY,
        includeTokens: ['val', 'validation', 'test', 'accuracy', 'acc', 'f1', 'auc', 'r2', 'score', 'precision', 'recall'],
        excludeTokens: ['train'],
        excludeKeys: DEFAULT_METRIC_EXCLUDE_KEYS,
      })
    : null;

  if (!hasLossSeries && trainFallbackKey && valFallbackKey && trainFallbackKey.toLowerCase() === valFallbackKey.toLowerCase()) {
    const lowered = valFallbackKey.toLowerCase();
    if (!lowered.includes('val') && !lowered.includes('validation') && !lowered.includes('test')) {
      valFallbackKey = null;
    }
  }

  const primaryTrainSeries = hasLossSeries
    ? trainLoss
    : (trainFallbackKey ? extractNumericSeries(metrics, trainFallbackKey) : []);
  const primaryValSeries = hasLossSeries
    ? valLoss
    : (valFallbackKey ? extractNumericSeries(metrics, valFallbackKey) : []);

  const trainMetricKey = hasLossSeries ? 'loss' : trainFallbackKey;
  const valMetricKey = hasLossSeries ? 'val_loss' : valFallbackKey;

  const trainMetricLabel = hasLossSeries
    ? 'Training Performance (Loss)'
    : `Training Performance (${formatMetricKeyLabel(trainMetricKey || valMetricKey)})`;
  const valMetricLabel = hasLossSeries
    ? 'Validation Loss'
    : (valMetricKey ? formatMetricKeyLabel(valMetricKey) : 'Validation Metric');

  const latestTrainMetric = hasLossSeries
    ? (overview?.latest_loss ?? latestSeriesValue(primaryTrainSeries))
    : latestSeriesValue(primaryTrainSeries);
  const latestValMetric = hasLossSeries
    ? (overview?.latest_val_loss ?? latestSeriesValue(primaryValSeries))
    : latestSeriesValue(primaryValSeries);

  const noPrimarySeriesMessage = hasLossSeries
    ? 'No loss metrics logged yet.'
    : 'No train/validation metrics logged yet.';

  const perplexity = extractNumericSeries(metrics, 'perplexity');
  const throughput = extractNumericSeries(metrics, 'tokens_per_sec').length
    ? extractNumericSeries(metrics, 'tokens_per_sec')
    : extractNumericSeries(metrics, 'throughput');
  const reward = extractNumericSeries(metrics, 'reward').length
    ? extractNumericSeries(metrics, 'reward')
    : extractNumericSeries(metrics, 'mean_reward');
  const klDiv = extractNumericSeries(metrics, 'kl_divergence').length
    ? extractNumericSeries(metrics, 'kl_divergence')
    : extractNumericSeries(metrics, 'kl');

  const commonChartOptions = {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
    grid: { left: 50, right: 30, bottom: 30, top: 30, containLabel: false },
    xAxis: {
      type: 'value',
      name: 'Step',
      nameLocation: 'middle',
      nameGap: 25,
      splitLine: { show: false },
      axisLabel: { color: textColor },
    },
    yAxis: {
      type: 'value',
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      axisLabel: { color: textColor },
    },
  };

  const trainLegendName = hasLossSeries ? 'Train Loss' : formatMetricKeyLabel(trainMetricKey || 'train_metric');
  const valLegendName = hasLossSeries ? 'Val Loss' : formatMetricKeyLabel(valMetricKey || 'validation_metric');
  const primaryLegendData = [
    ...(primaryTrainSeries.length > 0 ? [trainLegendName] : []),
    ...(primaryValSeries.length > 0 ? [valLegendName] : []),
  ];

  const primaryMetricOptions = {
    ...commonChartOptions,
    legend: {
      data: primaryLegendData,
      textStyle: { color: textColor },
      top: 0,
      right: 0,
    },
    series: [
      ...(primaryTrainSeries.length > 0
        ? [
            {
              name: trainLegendName,
              type: 'line',
              data: primaryTrainSeries,
              showSymbol: primaryTrainSeries.length === 1,
              symbolSize: primaryTrainSeries.length === 1 ? 9 : 5,
              smooth: true,
              lineStyle: { color: '#FDA481', width: primaryTrainSeries.length === 1 ? 0 : 2 },
              itemStyle: { color: '#FDA481' },
            },
          ]
        : []),
      ...(primaryValSeries.length > 0
        ? [
            {
              name: valLegendName,
              type: 'line',
              data: primaryValSeries,
              showSymbol: primaryValSeries.length === 1,
              symbolSize: primaryValSeries.length === 1 ? 9 : 5,
              smooth: true,
              lineStyle: { color: '#B4182D', width: primaryValSeries.length === 1 ? 0 : 2 },
              itemStyle: { color: '#B4182D' },
            },
          ]
        : []),
    ],
  };

  const lrAxisType = lrData.some((point) => point[1] > 0) ? 'log' : 'value';
  const lrOptions = {
    ...commonChartOptions,
    yAxis: {
      type: lrAxisType,
      name: 'Learning Rate',
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      axisLabel: { color: textColor },
    },
    series: [
      {
        name: 'LR',
        type: 'line',
        data: lrData,
        showSymbol: false,
        step: 'end',
        itemStyle: { color: '#FDA481' },
      },
    ],
  };

  const performanceOptions = {
    ...commonChartOptions,
    yAxis: { ...commonChartOptions.yAxis, name: 'Tokens/sec' },
    series: [
      {
        name: 'Throughput',
        type: 'line',
        data: throughput,
        showSymbol: false,
        smooth: true,
        itemStyle: { color: '#37415C' },
        areaStyle: { opacity: 0.1, color: '#37415C' },
      },
    ],
  };

  const advancedOptions = {
    ...commonChartOptions,
    legend: { textStyle: { color: textColor }, top: 0, right: 0 },
    series: [
      ...(perplexity.length
        ? [{ name: 'Perplexity', type: 'line', data: perplexity, showSymbol: false, smooth: true, itemStyle: { color: '#B4182D' } }]
        : []),
      ...(reward.length
        ? [{ name: 'Reward', type: 'line', data: reward, showSymbol: false, smooth: true, itemStyle: { color: '#FDA481' } }]
        : []),
      ...(klDiv.length
        ? [{ name: 'KL Div', type: 'line', data: klDiv, showSymbol: false, smooth: true, itemStyle: { color: '#37415C' } }]
        : []),
    ],
  };

  if (!metadata) return <div className="p-8 text-slate-500 dark:text-slate-400">Loading run data...</div>;

  const healthState = overview?.health_state || 'WARNING';
  const style = HEALTH_STYLES[healthState] || HEALTH_STYLES.WARNING;
  const StatusIcon = style.icon;
  const hasAlerts = alerts && alerts.length > 0;

  const currentStep = overview?.current_step ?? (metrics.length > 0 ? Math.max(...metrics.map((m) => m.step || 0)) : 0);
  const elapsedTime = overview?.elapsed_time_s ?? (metadata.start_time_epoch ? Date.now() / 1000 - metadata.start_time_epoch : null);
  const resolvedStatus = (overview?.status || metadata?.status || '').toLowerCase();
  const isTerminalRun = ['complete', 'completed', 'finished', 'failed', 'cancelled', 'interrupted'].includes(resolvedStatus);
  const etaText = isTerminalRun && overview?.eta_s === 0
    ? 'Complete'
    : (overview?.eta_s != null ? formatDuration(overview.eta_s) : (overview?.eta_reason || 'ETA unavailable'));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="h2 text-theme-text-primary">Global Experiment Overview</h1>
          <StatusBadge status={resolvedStatus} />
        </div>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border bg-white dark:bg-slate-900 shadow-sm ${style.border}`}>
          <StatusIcon className={`w-5 h-5 ${style.text}`} />
          <span className={`text-sm font-bold tracking-wide ${style.text}`}>SYSTEM: {healthState}</span>
        </div>
      </div>

      {hasAlerts && (
        <div className="space-y-3">
          {alerts.map((a, i) => (
            <div
              key={i}
              className={`p-4 rounded-lg flex items-start gap-3 border ${
                a.severity === 'high'
                  ? 'bg-red-50 dark:bg-red-500/10 border-red-200 dark:border-red-500/20'
                  : 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/20'
              }`}
            >
              <AlertTriangle
                className={`w-5 h-5 shrink-0 mt-0.5 ${
                  a.severity === 'high' ? 'text-red-600 dark:text-red-500' : 'text-amber-600 dark:text-amber-500'
                }`}
              />
              <div>
                <h4
                  className={`font-semibold text-sm ${
                    a.severity === 'high' ? 'text-red-700 dark:text-red-400' : 'text-amber-700 dark:text-amber-500'
                  }`}
                >
                  {a.title}
                </h4>
                <p
                  className={`text-sm mt-1 ${
                    a.severity === 'high' ? 'text-red-600 dark:text-red-400/80' : 'text-amber-600 dark:text-amber-500/80'
                  }`}
                >
                  {a.message}
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
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
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{formatDuration(elapsedTime)}</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Estimated Time Until Done</p>
            <Target className="w-4 h-4 text-amber-500" />
          </div>
          <p className="text-xl font-bold text-slate-900 dark:text-white">{etaText}</p>
          {overview?.eta_s != null && !isTerminalRun && (
            <span className="text-xs text-slate-500 dark:text-slate-400 mt-1">
              {overview?.eta_is_live ? 'Live ETA (monitor mode)' : 'Estimated from logged step timing'}
            </span>
          )}
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{trainMetricLabel}</p>
            <TrendingDown className="w-4 h-4 text-rose-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{formatFloat(latestTrainMetric, 5)}</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">{valMetricLabel}</p>
            <TrendingDown className="w-4 h-4 text-red-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{formatFloat(latestValMetric, 5)}</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Current Learning Rate</p>
            <Zap className="w-4 h-4 text-violet-500" />
          </div>
          <p className="text-2xl font-bold text-slate-900 dark:text-white">{formatFloat(overview?.current_lr, 6)}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card h-[380px] flex flex-col">
          <h3 className="h3 text-theme-text-primary mb-2">{trainMetricLabel}</h3>
          <div className="flex-1 min-h-0 relative">
            {primaryTrainSeries.length > 0 || primaryValSeries.length > 0 ? (
              <ReactECharts option={primaryMetricOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                {noPrimarySeriesMessage}
              </div>
            )}
          </div>
        </div>

        <div className="card h-[380px] flex flex-col">
          <h3 className="h3 text-theme-text-primary mb-2">Learning Rate Schedule</h3>
          <div className="flex-1 min-h-0 relative">
            {lrData.length > 0 ? (
              <ReactECharts option={lrOptions} style={{ height: '100%', width: '100%' }} />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-slate-400 text-sm">
                Learning rate stream unavailable for this run.
              </div>
            )}
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

      <div className="text-xs text-slate-500 dark:text-slate-400">
        Monitor mode: <span className="font-semibold">{overview?.monitor_enabled ? 'enabled' : 'disabled'}</span>
      </div>
    </div>
  );
}
