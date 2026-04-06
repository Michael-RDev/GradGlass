import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  ChevronDown,
  ChevronRight,
  Eye,
  EyeOff,
  Filter,
  LayoutGrid,
  ShieldAlert,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  SplitSquareVertical,
  Waves,
} from 'lucide-react';
import { fetchDataMonitor } from '../api';
import { EmptyState, LoadingSpinner, SeverityBadge, StatusBadge } from '../components/ui';
import { useTheme } from '../components/ThemeProvider';
import {
  buildInitialFilters,
  buildVisiblePanelOrder,
  filterChecks,
  filterCompositionPanels,
  filterSplitComparisonPanels,
  filterStageSnapshots,
  loadDataMonitorPreferences,
  movePanel,
  saveDataMonitorPreferences,
} from './dataMonitorUtils';

const PANEL_LABELS = {
  pipeline: 'Pipeline',
  composition: 'Composition',
  'split-comparisons': 'Split Comparisons',
  leakage: 'Leakage',
  recommendations: 'Recommendations',
};

const STATUS_STYLES = {
  passed: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-700 dark:text-emerald-400',
  warning: 'bg-amber-500/10 border-amber-500/30 text-amber-700 dark:text-amber-400',
  failed: 'bg-red-500/10 border-red-500/30 text-red-700 dark:text-red-400',
  unknown: 'bg-slate-500/10 border-slate-500/30 text-slate-700 dark:text-slate-400',
};

function formatValue(value, fallback = '-') {
  if (value == null || Number.isNaN(value)) return fallback;
  if (typeof value === 'number') return value.toLocaleString();
  return value;
}

function MetricPill({ label, value }) {
  return (
    <div className="rounded-xl border border-theme-border bg-theme-bg/50 px-4 py-2 hover:bg-theme-bg/80 transition-colors shadow-sm">
      <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted font-medium">{label}</p>
      <p className="mt-1 text-base font-bold text-theme-text-primary">{value}</p>
    </div>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <label className="flex flex-col gap-1.5 focus-within:text-theme-primary transition-colors">
      <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-theme-text-muted transition-colors group-focus-within:text-theme-primary">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-xl border border-theme-border bg-theme-surface px-4 py-2 text-sm font-medium text-theme-text-primary outline-none shadow-sm transition-all focus:border-theme-primary focus:ring-2 focus:ring-theme-primary/20 hover:border-theme-primary/50"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option === 'all' ? 'All' : option}
          </option>
        ))}
      </select>
    </label>
  );
}

function summarizeStageCoverage(snapshots, stage) {
  const matching = (snapshots || []).filter((snapshot) => snapshot.stage === stage);
  if (!matching.length) {
    return {
      sampleCount: null,
      observedCount: null,
      coverage: null,
      splits: [],
    };
  }
  const sampleCount = matching.reduce((sum, snapshot) => sum + (snapshot.sample_count || 0), 0);
  const observedCount = matching.reduce((sum, snapshot) => sum + (snapshot.observed_sample_count || 0), 0);
  return {
    sampleCount: sampleCount || null,
    observedCount: observedCount || 0,
    coverage: sampleCount > 0 ? observedCount / sampleCount : null,
    splits: [...new Set(matching.map((snapshot) => snapshot.split))].sort(),
  };
}

function StageGrid({ stageCards, snapshots }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
        {stageCards.map((card) => {
          const coverage = summarizeStageCoverage(snapshots, card.stage);
          const displaySampleCount = coverage.sampleCount ?? card.sample_count;
          const displaySplits = coverage.splits.length ? coverage.splits : card.splits;
          const observedSummary =
            coverage.observedCount == null
              ? null
              : coverage.coverage == null || coverage.coverage >= 0.9999
              ? `Analyzed ${formatValue(coverage.observedCount)} observed samples.`
              : `Analyzed ${formatValue(coverage.observedCount)} of ${formatValue(displaySampleCount)} samples (${(coverage.coverage * 100).toFixed(1)}% coverage).`;

          return (
            <div key={card.stage} className="rounded-2xl border border-theme-border/50 bg-theme-surface/70 backdrop-blur-md p-6 shadow-sm hover:shadow-md transition-all duration-300 hover:-translate-y-1 hover:border-theme-primary/40">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-base font-bold text-theme-text-primary">{card.title}</p>
                  <p className="mt-1 text-sm font-medium text-theme-text-muted">{displaySplits.length ? displaySplits.join(', ') : 'No recorded splits'}</p>
                </div>
                <span className={`badge shadow-sm px-2.5 py-0.5 ${STATUS_STYLES[card.status] || STATUS_STYLES.unknown}`}>
                  {card.status}
                </span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-3">
                <MetricPill label="Samples" value={formatValue(displaySampleCount)} />
                <MetricPill label="Splits" value={displaySplits.length || '-'} />
              </div>
              {observedSummary && (
                <p className="mt-4 text-xs leading-relaxed text-theme-text-muted">{observedSummary}</p>
              )}
            </div>
          );
        })}
      </div>

      <div className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
        <div className="overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="text-xs uppercase tracking-[0.18em] text-theme-text-muted">
              <tr>
                <th className="pb-3 pr-4">Stage</th>
                <th className="pb-3 pr-4">Split</th>
                <th className="pb-3 pr-4">Total Samples</th>
                <th className="pb-3 pr-4">Observed</th>
                <th className="pb-3 pr-4">Coverage</th>
                <th className="pb-3 pr-4">Dropped</th>
                <th className="pb-3 pr-4">Added</th>
                <th className="pb-3 pr-4">Missing</th>
                <th className="pb-3 pr-4">Labels</th>
                <th className="pb-3 pr-4">Latency</th>
                <th className="pb-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {snapshots.map((snapshot) => (
                <tr key={`${snapshot.stage}-${snapshot.split}`} className="border-t border-theme-border text-theme-text-secondary">
                  <td className="py-3 pr-4 font-medium text-theme-text-primary">{snapshot.stage_label}</td>
                  <td className="py-3 pr-4">{snapshot.split}</td>
                  <td className="py-3 pr-4">{formatValue(snapshot.sample_count)}</td>
                  <td className="py-3 pr-4">{formatValue(snapshot.observed_sample_count)}</td>
                  <td className="py-3 pr-4">{snapshot.sample_coverage == null ? '-' : `${(snapshot.sample_coverage * 100).toFixed(1)}%`}</td>
                  <td className="py-3 pr-4">{formatValue(snapshot.dropped_samples)}</td>
                  <td className="py-3 pr-4">{formatValue(snapshot.added_samples)}</td>
                  <td className="py-3 pr-4">
                    {snapshot.null_missing_rate == null ? '-' : `${(snapshot.null_missing_rate * 100).toFixed(1)}%`}
                  </td>
                  <td className="py-3 pr-4">{snapshot.label_availability}</td>
                  <td className="py-3 pr-4">{snapshot.latency_ms == null ? '-' : `${snapshot.latency_ms.toFixed(1)} ms`}</td>
                  <td className="py-3">
                    <StatusBadge status={snapshot.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function buildPieOption(series, theme) {
  const data = Object.entries(series || {}).map(([name, value]) => ({ name, value }));
  return {
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: theme === 'dark' ? 'rgba(255,255,255,0.8)' : '#37415C' } },
    series: [
      {
        type: 'pie',
        radius: ['42%', '72%'],
        data,
        label: { color: theme === 'dark' ? '#fff' : '#181A2F' },
        itemStyle: { borderRadius: 10, borderWidth: 2, borderColor: theme === 'dark' ? '#181A2F' : '#fff' },
      },
    ],
  };
}

function buildBarOption(data, theme) {
  const bins = data?.bins || [];
  const counts = data?.counts || [];
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 40, right: 20, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: bins,
      axisLabel: { color: theme === 'dark' ? 'rgba(255,255,255,0.75)' : '#37415C', rotate: bins.length > 4 ? 20 : 0 },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: theme === 'dark' ? 'rgba(255,255,255,0.75)' : '#37415C' },
      splitLine: { lineStyle: { color: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' } },
    },
    series: [
      {
        type: 'bar',
        data: counts,
        itemStyle: { color: '#FDA481', borderRadius: [8, 8, 0, 0] },
      },
    ],
  };
}

function buildCategoricalBarOption(series, theme) {
  const entries = Object.entries(series || {});
  return {
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 40, right: 20, top: 20, bottom: 60 },
    xAxis: {
      type: 'category',
      data: entries.map(([label]) => label),
      axisLabel: { color: theme === 'dark' ? 'rgba(255,255,255,0.75)' : '#37415C' },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: theme === 'dark' ? 'rgba(255,255,255,0.75)' : '#37415C', formatter: (value) => `${Math.round(value * 100)}%` },
      splitLine: { lineStyle: { color: theme === 'dark' ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)' } },
    },
    series: [
      {
        type: 'bar',
        data: entries.map(([, value]) => value),
        itemStyle: { color: '#37415C', borderRadius: [8, 8, 0, 0] },
      },
    ],
  };
}

function CompositionSection({ panels, theme }) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      {panels.map((panel) => (
        <div key={panel.id} className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-theme-text-primary">{panel.title}</p>
              <p className="mt-1 text-xs text-theme-text-muted">{panel.split ? `${panel.split} split` : 'All splits'}</p>
            </div>
            <span className="badge border-theme-border text-theme-text-secondary">{panel.type}</span>
          </div>

          {(panel.type === 'modality-breakdown' || panel.type === 'class-distribution') && (
            <ReactECharts
              option={panel.type === 'modality-breakdown' ? buildPieOption(panel.data.series, theme) : buildCategoricalBarOption(panel.data.series, theme)}
              style={{ height: 280, width: '100%' }}
            />
          )}

          {panel.type === 'histogram' && (
            <ReactECharts option={buildBarOption(panel.data, theme)} style={{ height: 280, width: '100%' }} />
          )}

          {panel.type === 'outlier-table' && (
            <div className="space-y-2">
              {panel.data.rows?.length ? panel.data.rows.map((row) => (
                <div key={row.feature} className="flex items-center justify-between rounded-xl border border-theme-border bg-theme-bg/50 px-3 py-2 text-sm">
                  <span className="font-medium text-theme-text-primary">{row.feature}</span>
                  <span className="text-theme-text-secondary">spread {row.spread}</span>
                </div>
              )) : <p className="text-sm text-theme-text-muted">No outlier features surfaced.</p>}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SplitComparisonsSection({ panels }) {
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      {panels.map((panel) => {
        const data = panel.data || {};
        const driftRows = data.numeric_drift_summary?.top_drifted_features || [];
        return (
          <div key={panel.id} className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-theme-text-primary">{panel.title}</p>
                <p className="mt-1 text-xs text-theme-text-muted">{data.summary || 'Split comparison summary'}</p>
              </div>
              <SplitSquareVertical className="h-5 w-5 text-theme-primary" />
            </div>

            <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
              <MetricPill label="Max Label Diff" value={data.label_distribution_diff?.max_abs_diff == null ? '-' : `${(data.label_distribution_diff.max_abs_diff * 100).toFixed(1)}%`} />
              <MetricPill label="Modality Signals" value={Object.keys(data.modality_diff || {}).length || '-'} />
              <MetricPill label="Drifted Features" value={driftRows.length || '-'} />
            </div>

            {driftRows.length > 0 && (
              <div className="mt-4 space-y-2">
                {driftRows.map((row) => (
                  <div key={row.feature} className="flex items-center justify-between rounded-xl border border-theme-border bg-theme-bg/50 px-3 py-2 text-sm">
                    <span className="font-medium text-theme-text-primary">{row.feature}</span>
                    <span className="text-theme-text-secondary">mean diff {row.mean_diff}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function LeakageSection({ checks }) {
  if (!checks.length) {
    return <EmptyState icon={ShieldAlert} title="No checks matched the current filters" description="Adjust the filters or record more splits/stages." />;
  }

  return (
    <div className="space-y-4">
      {checks.map((check) => (
        <details key={check.name} className="group rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
          <summary className="flex cursor-pointer list-none items-start justify-between gap-4">
            <div className="flex min-w-0 items-start gap-3">
              {check.status === 'passed' ? (
                <ShieldCheck className="mt-0.5 h-5 w-5 text-emerald-500" />
              ) : (
                <ShieldAlert className="mt-0.5 h-5 w-5 text-theme-secondary" />
              )}
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-theme-text-primary">{check.name}</p>
                  <SeverityBadge severity={check.severity} />
                  <StatusBadge status={check.status} />
                </div>
                <p className="mt-2 text-sm text-theme-text-secondary">{check.summary}</p>
                {check.recommendation && (
                  <p className="mt-2 text-sm text-theme-primary">{check.recommendation}</p>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 text-theme-text-muted">
              <span className="text-xs">{check.duration_ms?.toFixed ? `${check.duration_ms.toFixed(0)} ms` : '-'}</span>
              <ChevronRight className="h-4 w-4 transition-transform group-open:rotate-90" />
            </div>
          </summary>

          <div className="mt-4 border-t border-theme-border pt-4">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Evidence</p>
                {check.evidence?.length ? check.evidence.map((item, index) => (
                  <div key={`${check.name}-evidence-${index}`} className="rounded-xl border border-theme-border bg-theme-bg/50 p-3 text-xs text-theme-text-secondary">
                    <pre className="whitespace-pre-wrap break-all">{JSON.stringify(item, null, 2)}</pre>
                  </div>
                )) : <p className="text-sm text-theme-text-muted">No evidence payload attached.</p>}
              </div>

              <div className="space-y-2">
                <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Debug Payload</p>
                <div className="rounded-xl border border-theme-border bg-theme-bg/50 p-3 text-xs text-theme-text-secondary">
                  <pre className="whitespace-pre-wrap break-all">{JSON.stringify({ payload: check.payload, metrics: check.metrics }, null, 2)}</pre>
                </div>
              </div>
            </div>
          </div>
        </details>
      ))}
    </div>
  );
}

function RecommendationsSection({ recommendations }) {
  if (!recommendations.length) {
    return <EmptyState icon={Sparkles} title="No follow-up recommendations" description="Warnings and failures will surface prioritized next steps here." />;
  }
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      {recommendations.map((item) => (
        <div key={`${item.title}-${item.score}`} className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-theme-text-primary">{item.title}</p>
              <p className="mt-2 text-sm text-theme-text-secondary">{item.summary}</p>
            </div>
            <SeverityBadge severity={item.severity} />
          </div>
          <div className="mt-4 grid grid-cols-3 gap-3">
            <MetricPill label="Confidence" value={`${Math.round((item.confidence || 0) * 100)}%`} />
            <MetricPill label="Score" value={item.score?.toFixed ? item.score.toFixed(1) : item.score} />
            <MetricPill label="Splits" value={item.affected_splits?.length ? item.affected_splits.join(', ') : '-'} />
          </div>
          {item.next_steps?.length > 0 && (
            <div className="mt-4 rounded-xl border border-theme-border bg-theme-bg/50 p-3 text-sm text-theme-text-secondary">
              {item.next_steps.join(' ')}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function PanelShell({ id, title, icon: Icon, collapsed, onToggle, explanation, children }) {
  return (
    <section className="glass-panel">
      <button className="group flex w-full items-center justify-between gap-4 text-left outline-none" onClick={onToggle}>
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-theme-primary/10 text-theme-primary transition-transform group-hover:scale-105">
            <Icon className="h-6 w-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-theme-text-primary transition-colors group-hover:text-theme-primary">{title}</h2>
          </div>
        </div>
        <div className="flex items-center justify-center h-8 w-8 rounded-full bg-theme-bg/50 transition-colors group-hover:bg-theme-primary/10">
          {collapsed ? <ChevronRight className="h-5 w-5 text-theme-text-muted group-hover:text-theme-primary" /> : <ChevronDown className="h-5 w-5 text-theme-text-muted group-hover:text-theme-primary" />}
        </div>
      </button>
      {!collapsed && (
        <div className="mt-8">
          {explanation && (
            <div className="rounded-2xl border border-theme-primary/30 bg-theme-primary/5 p-5 mb-6">
              <h3 className="text-sm font-bold text-theme-primary mb-2">🎓 What to look for: {title}</h3>
              <p className="text-sm text-theme-text-secondary w-full max-w-4xl">{explanation}</p>
            </div>
          )}
          {children}
        </div>
      )}
    </section>
  );
}

export default function Data({ routeMode = 'data' }) {
  const { runId } = useParams();
  const { theme } = useTheme();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [preferences, setPreferences] = useState(() => loadDataMonitorPreferences());
  const [filters, setFilters] = useState(() => buildInitialFilters(routeMode, loadDataMonitorPreferences().defaultFilters));
  const [customizeOpen, setCustomizeOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    if (!runId) return undefined;
    setLoading(true);
    setError(null);
    fetchDataMonitor(runId)
      .then((payload) => {
        if (!cancelled) {
          setReport(payload);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setReport(null);
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  useEffect(() => {
    saveDataMonitorPreferences({
      ...preferences,
      defaultFilters: filters,
    });
  }, [preferences, filters]);

  const dashboard = report?.dashboard || {};
  const summary = dashboard.summary || {};
  const visiblePanelOrder = useMemo(() => buildVisiblePanelOrder(preferences, routeMode), [preferences, routeMode]);
  const stageSnapshots = useMemo(() => filterStageSnapshots(report?.pipeline?.snapshots || [], filters), [report, filters]);
  const stageCards = useMemo(() => dashboard.stage_cards || [], [dashboard]);
  const compositionPanels = useMemo(() => filterCompositionPanels(dashboard.composition_panels || [], filters), [dashboard, filters]);
  const splitComparisonPanels = useMemo(() => filterSplitComparisonPanels(dashboard.split_comparison_panels || [], filters), [dashboard, filters]);
  const checks = useMemo(() => filterChecks(report?.checks || [], filters), [report, filters]);
  const recommendations = useMemo(() => report?.recommendations || [], [report]);
  const filterOptions = dashboard.filters || { splits: [], stages: [], categories: [], severities: [], statuses: [], modalities: [] };

  const headerStatus = summary.overall_status || report?.metadata?.overall_status || 'unknown';

  const panelContent = {
    pipeline: (
      <StageGrid stageCards={stageCards} snapshots={stageSnapshots} />
    ),
    composition: (
      <CompositionSection panels={compositionPanels} theme={theme} />
    ),
    'split-comparisons': (
      <SplitComparisonsSection panels={splitComparisonPanels} />
    ),
    leakage: (
      <LeakageSection checks={checks} />
    ),
    recommendations: (
      <RecommendationsSection recommendations={recommendations} />
    ),
  };

  const panelIcons = {
    pipeline: LayoutGrid,
    composition: Waves,
    'split-comparisons': SplitSquareVertical,
    leakage: ShieldAlert,
    recommendations: Sparkles,
  };

  if (loading) return <LoadingSpinner />;

  if (error || !report) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="Dataset monitor unavailable"
        description={error || 'Generate a dataset monitor report with run.monitor_dataset(...).finalize() or run.check_leakage(...).'}
      />
    );
  }

  const explanations = {
    pipeline: 'Look closely at Stage Coverage. Dropped samples or low coverage might indicate data parsing errors before training even begins. Make sure no unexpected skips happen in your dataset loaders.',
    composition: 'Check the class distributions. A heavily imbalanced dataset will cause your model to ignore minority classes. Ensure histograms match your expectations for realistic data.',
    'split-comparisons': 'Watch out for distribution drift between train and validation splits. If val split is significantly different from train, your metrics will misrepresent real model capability.',
    leakage: 'Leakage occurs when target information sneaks into training inputs. If a check fails here, your model is likely memorizing instead of learning.',
    recommendations: 'Follow these prioritized action items to resolve critical pipeline errors and performance degradation.'
  };

  return (
    <div className="space-y-8">
      <div className="glass-panel">
        <div className="flex flex-col gap-8 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="flex flex-wrap items-center gap-4">
              <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-theme-text-primary to-theme-primary">
                Dataset & Pipeline Monitor
              </h1>
              <span className={`badge px-3 py-1 font-bold uppercase tracking-wider backdrop-blur-md shadow-sm ${STATUS_STYLES[headerStatus] || STATUS_STYLES.unknown}`}>
                {headerStatus}
              </span>
              {routeMode === 'leakage' && (
                <span className="badge border-theme-primary/40 bg-theme-primary/10 text-theme-primary px-3 py-1 font-bold uppercase tracking-wider backdrop-blur-md shadow-sm">Leakage focus</span>
              )}
            </div>
            <p className="mt-6 max-w-2xl text-base leading-relaxed text-theme-text-secondary">
              Extensible dataset observability across raw data, transformed stages, splits, tokenization, loaders, and leakage diagnostics. Monitor your data health right alongside training metrics.
            </p>
            <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricPill label="Task" value={report.metadata?.task || '-'} />
              <MetricPill label="Stages" value={summary.recorded_stage_count ?? report.metadata?.recorded_stage_count ?? '-'} />
              <MetricPill label="Checks" value={summary.total_checks ?? report.metadata?.total_checks ?? '-'} />
              <MetricPill label="Warnings / Failures" value={`${summary.warning_checks || 0} / ${summary.failed_checks || 0}`} />
            </div>
          </div>

          <div className="flex flex-col gap-3 xl:min-w-[360px]">
            <div className="flex items-center gap-3">
              <button className="btn-primary inline-flex items-center gap-2" onClick={() => setCustomizeOpen((value) => !value)}>
                <SlidersHorizontal className="h-4 w-4" />
                Customize View
              </button>
              <button
                className="rounded-xl border border-theme-border px-4 py-2 text-sm text-theme-text-secondary transition-colors hover:bg-theme-bg"
                onClick={() => setFilters(buildInitialFilters(routeMode))}
              >
                Reset Filters
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
              <SelectField label="Split" value={filters.split} onChange={(value) => setFilters((prev) => ({ ...prev, split: value }))} options={['all', ...(filterOptions.splits || [])]} />
              <SelectField label="Stage" value={filters.stage} onChange={(value) => setFilters((prev) => ({ ...prev, stage: value }))} options={['all', ...(filterOptions.stages || [])]} />
              <SelectField label="Category" value={filters.category} onChange={(value) => setFilters((prev) => ({ ...prev, category: value }))} options={['all', ...(filterOptions.categories || [])]} />
              <SelectField label="Severity" value={filters.severity} onChange={(value) => setFilters((prev) => ({ ...prev, severity: value }))} options={['all', ...(filterOptions.severities || [])]} />
              <SelectField label="Status" value={filters.status} onChange={(value) => setFilters((prev) => ({ ...prev, status: value }))} options={['all', ...(filterOptions.statuses || [])]} />
              <SelectField label="Modality" value={filters.modality} onChange={(value) => setFilters((prev) => ({ ...prev, modality: value }))} options={['all', ...(filterOptions.modalities || [])]} />
            </div>
          </div>
        </div>

        {customizeOpen && (
          <div className="mt-6 glass-card">
            <div className="mb-4 flex items-center gap-2">
              <Filter className="h-4 w-4 text-theme-primary" />
              <p className="text-sm font-semibold text-theme-text-primary">Panel Layout</p>
            </div>
            <div className="space-y-3">
              {preferences.panelOrder.map((panelId) => (
                <div key={panelId} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-theme-border bg-theme-surface px-3 py-3 hover:border-theme-primary/30 transition-colors">
                  <div>
                    <p className="text-sm font-medium text-theme-text-primary">{PANEL_LABELS[panelId]}</p>
                    <p className="text-xs text-theme-text-muted">Persisted locally for this browser.</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      className="rounded-lg border border-theme-border p-2 text-theme-text-secondary hover:bg-theme-bg transition-colors"
                      onClick={() => setPreferences((prev) => ({ ...prev, panelOrder: movePanel(prev.panelOrder, panelId, 'up') }))}
                    >
                      <ArrowUp className="h-4 w-4" />
                    </button>
                    <button
                      className="rounded-lg border border-theme-border p-2 text-theme-text-secondary hover:bg-theme-bg transition-colors"
                      onClick={() => setPreferences((prev) => ({ ...prev, panelOrder: movePanel(prev.panelOrder, panelId, 'down') }))}
                    >
                      <ArrowDown className="h-4 w-4" />
                    </button>
                    <button
                      className="rounded-lg border border-theme-border p-2 text-theme-text-secondary hover:bg-theme-bg transition-colors"
                      onClick={() => setPreferences((prev) => ({
                        ...prev,
                        panelVisibility: {
                          ...prev.panelVisibility,
                          [panelId]: !prev.panelVisibility[panelId],
                        },
                      }))}
                    >
                      {preferences.panelVisibility[panelId] === false ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {visiblePanelOrder.map((panelId) => (
        <PanelShell
          key={panelId}
          id={panelId}
          title={PANEL_LABELS[panelId]}
          icon={panelIcons[panelId]}
          explanation={explanations[panelId]}
          collapsed={preferences.collapsedSections[panelId]}
          onToggle={() => setPreferences((prev) => ({
            ...prev,
            collapsedSections: {
              ...prev.collapsedSections,
              [panelId]: !prev.collapsedSections[panelId],
            },
          }))}
        >
          {panelContent[panelId]}
        </PanelShell>
      ))}
    </div>
  );
}
