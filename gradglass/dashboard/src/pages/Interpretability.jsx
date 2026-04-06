import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  AlertCircle,
  BrainCircuit,
  Lightbulb,
  ScanSearch,
  Sparkles,
  Waves,
} from 'lucide-react';
import { fetchAnalysis, fetchPredictions, fetchSaliency, fetchShap, fetchLime } from '../api';
import { buildAttentionHeatmapOption, buildStructuredSaliencyOption, formatInterpretabilityValue, HeatmapGrid } from '../components/SaliencyShared';
import { EmptyState, ErrorMessage, LoadingSpinner, SeverityBadge, StatusBadge } from '../components/ui';
import { useTheme } from '../components/ThemeProvider';
import { createInterpretabilityViewModel } from './interpretabilityUtils';

function MetricCard({ label, value, hint }) {
  return (
    <div className="rounded-2xl border border-theme-border bg-theme-surface/70 px-4 py-3 shadow-sm">
      <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">{label}</p>
      <p className="mt-1 text-lg font-bold text-theme-text-primary">{value}</p>
      {hint && <p className="mt-1 text-xs text-theme-text-muted">{hint}</p>}
    </div>
  );
}

function InsightCard({ card }) {
  return (
    <div className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold text-theme-text-primary">{card.title}</p>
          <p className="mt-2 text-sm leading-relaxed text-theme-text-secondary">{card.summary}</p>
        </div>
        <div className="flex flex-col items-end gap-2">
          <StatusBadge status={card.status} />
          <SeverityBadge severity={card.severity} />
        </div>
      </div>
      {card.recommendation && (
        <div className="mt-4 rounded-xl border border-theme-border bg-theme-bg/50 p-3 text-sm text-theme-text-secondary">
          {card.recommendation}
        </div>
      )}
    </div>
  );
}

function AttributionSection({ saliency, cards, textColor, gridColor }) {
  const saliencyAvailable = saliency?.available === true;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        {cards.map((card) => (
          <InsightCard key={card.id} card={card} />
        ))}
      </div>

      <div className="rounded-[28px] border border-theme-border bg-theme-surface p-6 shadow-md">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold text-theme-text-primary">Live Attribution</h2>
            <p className="mt-2 text-sm text-theme-text-secondary">
              Real probe artifacts from the latest captured run step. GradGlass shows saliency for vision and structured tensors when those artifacts exist.
            </p>
          </div>
          {saliencyAvailable && (
            <div className="flex items-center gap-2">
              <StatusBadge status="passed" />
              <span className="rounded-full border border-theme-border px-3 py-1 text-xs text-theme-text-secondary">
                Step {saliency.step ?? '—'} · {saliency.modality}
              </span>
            </div>
          )}
        </div>

        {!saliencyAvailable ? (
          <div className="mt-6 rounded-2xl border border-dashed border-theme-border bg-theme-bg/40 px-6 py-10">
            <p className="text-sm font-medium text-theme-text-primary">Saliency is not available for this run yet.</p>
            <p className="mt-2 text-sm text-theme-text-secondary">
              {saliency?.reason || "Re-run with `run.watch(..., saliency='auto', gradients='summary')` to capture attribution probes."}
            </p>
          </div>
        ) : saliency.modality === 'vision' ? (
          <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-2">
            {saliency.samples.map((sample) => (
              <div key={sample.index} className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 shadow-sm">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-theme-text-primary">Sample {sample.index + 1}</p>
                    <p className="text-xs text-theme-text-muted">
                      Target {formatInterpretabilityValue(sample.target)} · Prediction {formatInterpretabilityValue(sample.prediction)}
                    </p>
                  </div>
                  {sample.confidence != null && (
                    <span className="rounded-full border border-rose-500/20 bg-rose-500/10 px-2.5 py-1 text-xs text-rose-400">
                      {(Number(sample.confidence) * 100).toFixed(1)}% confidence
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="mb-2 text-xs uppercase tracking-[0.18em] text-theme-text-muted">Input</p>
                    <HeatmapGrid values={sample.input} tone="input" />
                  </div>
                  <div>
                    <p className="mb-2 text-xs uppercase tracking-[0.18em] text-theme-text-muted">Saliency</p>
                    <HeatmapGrid values={sample.saliency} tone="saliency" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : saliency.modality === 'structured' ? (
          <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)]">
            <div className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 shadow-sm">
              <div className="mb-3">
                <p className="text-sm font-semibold text-theme-text-primary">Most Influential Features</p>
                <p className="mt-1 text-xs text-theme-text-muted">Mean absolute saliency across the latest probe examples.</p>
              </div>
              <div className="h-[360px]">
                <ReactECharts option={buildStructuredSaliencyOption(saliency.feature_importance, textColor, gridColor)} style={{ height: '100%', width: '100%' }} />
              </div>
            </div>
            <div className="space-y-3">
              {saliency.samples.slice(0, 4).map((sample) => (
                <div key={sample.index} className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 shadow-sm">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-theme-text-primary">Sample {sample.index + 1}</p>
                      <p className="text-xs text-theme-text-muted">
                        Target {formatInterpretabilityValue(sample.target)} · Prediction {formatInterpretabilityValue(sample.prediction)}
                      </p>
                    </div>
                    {sample.confidence != null && (
                      <span className="rounded-full border border-rose-500/20 bg-rose-500/10 px-2.5 py-1 text-xs text-rose-400">
                        {(Number(sample.confidence) * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                  <div className="mt-3 rounded-xl border border-theme-border bg-theme-surface px-3 py-2 text-xs text-theme-text-secondary">
                    <p className="font-medium text-theme-text-primary">Input Snapshot</p>
                    <p className="mt-1 truncate">{sample.input.slice(0, 8).map((value) => Number(value).toFixed(3)).join(', ')}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="mt-6 rounded-2xl border border-dashed border-theme-border bg-theme-bg/40 px-6 py-10 text-sm text-theme-text-secondary">
            Saliency was captured, but this payload format is not supported by the unified view yet.
          </div>
        )}
      </div>
    </div>
  );
}

function HardExamplesSection({ predictions, hardExamples }) {
  const latestStep = predictions?.predictions?.at(-1)?.step;
  const hasPredictionData = Array.isArray(predictions?.predictions) && predictions.predictions.length > 0;

  if (!hasPredictionData) {
    return (
      <EmptyState
        icon={AlertCircle}
        title="No prediction probes available"
        description="Log predictions with `run.log_batch(x, y, y_pred)` to surface hard examples and unstable samples."
      />
    );
  }

  if (!hardExamples.length) {
    return (
      <div className="rounded-[28px] border border-theme-border bg-theme-surface p-6 shadow-md">
        <p className="text-lg font-semibold text-theme-text-primary">No hard examples surfaced from the latest logged probe.</p>
        <p className="mt-2 text-sm text-theme-text-secondary">
          Step {latestStep ?? '—'} did not contain recent misclassifications or unstable label flips in the tracked sample window.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="rounded-[28px] border border-theme-border bg-theme-surface p-6 shadow-md">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-bold text-theme-text-primary">Hard Examples</h2>
            <p className="mt-2 text-sm text-theme-text-secondary">
              Ranked from the latest logged prediction probe. Confident misses are prioritized first, then recent regressions and unstable label flips.
            </p>
          </div>
          <span className="rounded-full border border-theme-border px-3 py-1 text-xs text-theme-text-secondary">
            Step {latestStep ?? '—'} · {hardExamples.length} surfaced
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        {hardExamples.map((example) => (
          <div key={example.index} className="rounded-2xl border border-theme-border bg-theme-surface p-5 shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm font-semibold text-theme-text-primary">Sample #{example.index}</p>
                <p className="mt-2 text-sm text-theme-text-secondary">{example.reason}</p>
              </div>
              <StatusBadge status={!example.isCorrect ? 'warning' : 'passed'} />
            </div>

            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-xl border border-theme-border bg-theme-bg/50 p-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Target</p>
                <p className="mt-1 text-lg font-bold text-theme-text-primary">{formatInterpretabilityValue(example.trueLabel)}</p>
              </div>
              <div className="rounded-xl border border-theme-border bg-theme-bg/50 p-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Prediction</p>
                <p className={`mt-1 text-lg font-bold ${example.isCorrect ? 'text-emerald-500' : 'text-rose-500'}`}>
                  {formatInterpretabilityValue(example.prediction)}
                </p>
              </div>
            </div>

            <div className="mt-4 space-y-3 text-sm text-theme-text-secondary">
              {example.confidence != null && (
                <div>
                  <div className="mb-1 flex items-center justify-between gap-3 text-xs uppercase tracking-[0.18em] text-theme-text-muted">
                    <span>Confidence</span>
                    <span>{(Number(example.confidence) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-theme-bg">
                    <div className="h-full rounded-full bg-gradient-to-r from-orange-400 to-rose-500" style={{ width: `${Math.max(4, Number(example.confidence) * 100)}%` }} />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                <div className="rounded-xl border border-theme-border bg-theme-bg/40 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Previous Prediction</p>
                  <p className="mt-1 text-theme-text-primary">{formatInterpretabilityValue(example.prevPrediction)}</p>
                </div>
                <div className="rounded-xl border border-theme-border bg-theme-bg/40 px-3 py-2">
                  <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Shift</p>
                  <p className="mt-1 text-theme-text-primary">
                    {example.changed ? 'Label changed' : 'Stable label'}
                    {example.confidenceDelta != null ? ` · ${(example.confidenceDelta * 100).toFixed(1)} pts` : ''}
                  </p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function resolveAttentionMatrix(attention) {
  if (!attention) return { tokens: [], matrix: null };
  if (Array.isArray(attention.tokens) && Array.isArray(attention.matrix)) {
    return { tokens: attention.tokens, matrix: attention.matrix };
  }
  if (Array.isArray(attention.labels) && Array.isArray(attention.weights)) {
    return { tokens: attention.labels, matrix: attention.weights };
  }
  const firstHead = attention.heads?.[0];
  if (Array.isArray(firstHead?.tokens) && Array.isArray(firstHead?.matrix)) {
    return { tokens: firstHead.tokens, matrix: firstHead.matrix };
  }
  const firstLayer = attention.layers?.[0];
  if (Array.isArray(firstLayer?.tokens) && Array.isArray(firstLayer?.matrix)) {
    return { tokens: firstLayer.tokens, matrix: firstLayer.matrix };
  }
  return { tokens: [], matrix: null };
}

function AttentionSection({ attention, textColor, theme }) {
  const { tokens, matrix } = resolveAttentionMatrix(attention);

  if (tokens.length && matrix) {
    return (
      <div className="rounded-[28px] border border-theme-border bg-theme-surface p-6 shadow-md">
        <div>
          <h2 className="text-xl font-bold text-theme-text-primary">Attention Map</h2>
          <p className="mt-2 text-sm text-theme-text-secondary">Attention artifacts were detected for this run, so the page promoted a dedicated attention view.</p>
        </div>
        <div className="mt-6 h-[460px]">
          <ReactECharts option={buildAttentionHeatmapOption(tokens, matrix, textColor, theme)} style={{ height: '100%', width: '100%' }} />
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-[28px] border border-theme-border bg-theme-surface p-6 shadow-md">
      <h2 className="text-xl font-bold text-theme-text-primary">Attention Payload</h2>
      <p className="mt-2 text-sm text-theme-text-secondary">Attention artifacts exist for this run, but the payload shape is not yet standardized enough for a richer renderer.</p>
      <div className="mt-6 rounded-2xl border border-theme-border bg-theme-bg/40 p-4 text-xs text-theme-text-secondary">
        <pre className="whitespace-pre-wrap break-all">{JSON.stringify(attention, null, 2)}</pre>
      </div>
    </div>
  );
}

export default function Interpretability() {
  const { runId } = useParams();
  const { theme } = useTheme();
  const [payloads, setPayloads] = useState({ saliency: null, predictions: null, analysis: null, shap: null, lime: null });
  const [partialErrors, setPartialErrors] = useState({});
  const [loading, setLoading] = useState(true);
  const [fatalError, setFatalError] = useState(null);
  const [activeTab, setActiveTab] = useState('attribution');

  useEffect(() => {
    let cancelled = false;
    if (!runId) return undefined;

    setLoading(true);
    setFatalError(null);

    Promise.allSettled([fetchSaliency(runId), fetchPredictions(runId), fetchAnalysis(runId), fetchShap(runId), fetchLime(runId)])
      .then(([saliencyResult, predictionsResult, analysisResult, shapResult, limeResult]) => {
        if (cancelled) return;

        const nextPayloads = {
          saliency: saliencyResult.status === 'fulfilled' ? saliencyResult.value : null,
          predictions: predictionsResult.status === 'fulfilled' ? predictionsResult.value : null,
          analysis: analysisResult.status === 'fulfilled' ? analysisResult.value : null,
          shap: shapResult.status === 'fulfilled' ? shapResult.value : null,
          lime: limeResult.status === 'fulfilled' ? limeResult.value : null,
        };
        const nextErrors = {
          saliency: saliencyResult.status === 'rejected' ? saliencyResult.reason?.message || String(saliencyResult.reason) : null,
          predictions: predictionsResult.status === 'rejected' ? predictionsResult.reason?.message || String(predictionsResult.reason) : null,
          analysis: analysisResult.status === 'rejected' ? analysisResult.reason?.message || String(analysisResult.reason) : null,
          shap: shapResult.status === 'rejected' ? shapResult.reason?.message || String(shapResult.reason) : null,
          lime: limeResult.status === 'rejected' ? limeResult.reason?.message || String(limeResult.reason) : null,
        };

        if (!nextPayloads.saliency && !nextPayloads.predictions && !nextPayloads.analysis && !nextPayloads.shap && !nextPayloads.lime) {
          setFatalError(nextErrors.analysis || nextErrors.saliency || nextErrors.predictions || 'Interpretability artifacts could not be loaded.');
          setPayloads({ saliency: null, predictions: null, analysis: null, shap: null, lime: null });
          setPartialErrors({});
          return;
        }

        setPayloads(nextPayloads);
        setPartialErrors(Object.fromEntries(Object.entries(nextErrors).filter(([, value]) => value)));
      })
      .catch((error) => {
        if (!cancelled) {
          setFatalError(error.message);
          setPayloads({ saliency: null, predictions: null, analysis: null });
          setPartialErrors({});
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

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.10)' : 'rgba(15, 23, 42, 0.08)';

  const viewModel = useMemo(() => createInterpretabilityViewModel(payloads), [payloads]);
  const tabs = viewModel.tabs;
  const saliencyAvailable = payloads.saliency?.available === true;
  const activeSignals = viewModel.cards.filter((card) => card.status !== 'skip').length;

  useEffect(() => {
    if (!tabs.some((tab) => tab.id === activeTab) && tabs.length) {
      setActiveTab(tabs[0].id);
    }
  }, [tabs, activeTab]);

  if (loading) return <LoadingSpinner />;
  if (fatalError) return <ErrorMessage message={fatalError} />;

  return (
    <div className="space-y-8">
      <div className="rounded-[32px] border border-theme-border/60 bg-theme-surface/80 p-8 shadow-md">
        <div className="flex flex-col gap-8 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-3xl">
            <div className="flex flex-wrap items-center gap-4">
              <h1 className="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-theme-text-primary to-theme-primary">
                Interpretability & Debugging
              </h1>
              <StatusBadge status={saliencyAvailable ? 'passed' : 'warning'} />
            </div>
            <p className="mt-4 max-w-2xl text-base leading-relaxed text-theme-text-secondary">
              Unified interpretability across real saliency probes, prediction failures, and post-run attribution diagnostics. Attention only appears when a run actually exposes attention artifacts.
            </p>
            <div className="mt-8 grid grid-cols-2 gap-4 md:grid-cols-4">
              <MetricCard label="Attribution" value={saliencyAvailable ? 'Available' : 'Missing'} hint={payloads.saliency?.modality || 'needs saliency probes'} />
              <MetricCard label="Hard Examples" value={viewModel.hardExamples.length} hint="latest logged probe" />
              <MetricCard label="Signals" value={activeSignals} hint="analysis cards with data" />
              <MetricCard label="Prediction Probes" value={payloads.predictions?.predictions?.length || 0} hint="logged batches" />
            </div>
          </div>

          <div className="min-w-[320px] rounded-2xl border border-theme-border bg-theme-bg/50 p-4">
            <p className="text-[11px] uppercase tracking-[0.18em] text-theme-text-muted">Artifacts</p>
            <div className="mt-3 space-y-3 text-sm text-theme-text-secondary">
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2"><ScanSearch className="h-4 w-4 text-theme-primary" /> Saliency</span>
                <StatusBadge status={payloads.saliency?.available ? 'passed' : 'warning'} />
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2"><Sparkles className="h-4 w-4 text-theme-primary" /> Predictions</span>
                <StatusBadge status={payloads.predictions?.predictions?.length ? 'passed' : 'warning'} />
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2"><BrainCircuit className="h-4 w-4 text-theme-primary" /> Analysis</span>
                <StatusBadge status={payloads.analysis?.tests?.results?.length ? 'passed' : 'warning'} />
              </div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2"><Waves className="h-4 w-4 text-theme-primary" /> Attention</span>
                <StatusBadge status={viewModel.attention ? 'passed' : 'skip'} />
              </div>
            </div>
          </div>
        </div>

        {Object.keys(partialErrors).length > 0 && (
          <div className="mt-6 rounded-2xl border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-700 dark:text-amber-400">
            <div className="flex items-start gap-3">
              <Lightbulb className="mt-0.5 h-4 w-4 shrink-0" />
              <p>
                Some interpretability sections are degraded because one or more artifact requests failed.
                {' '}
                {Object.entries(partialErrors).map(([key, value]) => `${key}: ${value}`).join(' | ')}
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap border-b border-theme-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`inline-flex items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'border-theme-primary text-theme-primary'
                : 'border-transparent text-theme-text-secondary hover:text-theme-text-primary'
            }`}
          >
            {tab.id === 'attribution' ? <ScanSearch className="h-4 w-4" /> : null}
            {tab.id === 'hard-examples' ? <AlertCircle className="h-4 w-4" /> : null}
            {tab.id === 'attention' ? <BrainCircuit className="h-4 w-4" /> : null}
            {tab.label}
          </button>
        ))}
      </div>

      {activeTab === 'attribution' && (
        <AttributionSection saliency={payloads.saliency} cards={viewModel.cards} textColor={textColor} gridColor={gridColor} />
      )}
        {activeTab === 'hard-examples' && (
        <HardExamplesSection predictions={payloads.predictions} hardExamples={viewModel.hardExamples} />
      )}
      {activeTab === 'attention' && viewModel.attention && (
        <AttentionSection attention={viewModel.attention} textColor={textColor} theme={theme} />
      )}
      {activeTab === 'shap' && (
        <div className="glass-panel space-y-6">
          <div className="rounded-2xl border border-theme-primary/30 bg-theme-primary/5 p-5">
            <h3 className="text-sm font-bold text-theme-primary mb-2">🎓 What to look for: SHAP Analysis</h3>
            <p className="text-sm text-theme-text-secondary">Look for high magnitude SHAP values (both positive and negative), as they indicate strong feature influence on the model's output. Global feature importance helps you debug whether your neural net relies on spurious correlations or expected signals.</p>
          </div>
          <div>
            <h2 className="text-xl font-bold text-theme-text-primary">Global Feature Importance (SHAP)</h2>
            <p className="mt-2 text-sm text-theme-text-secondary">{viewModel.shap?.message || 'Aggregated SHAP values across probes.'}</p>
          </div>
          <div className="h-[400px] mt-4">
            {viewModel.shap?.summary_plot ? (
              <ReactECharts
                option={{
                  tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
                  grid: { left: '20%', right: '5%', top: '5%', bottom: '10%' },
                  xAxis: { type: 'value', axisLabel: { color: textColor }, splitLine: { lineStyle: { color: gridColor, type: 'dashed' } } },
                  yAxis: { type: 'category', data: viewModel.shap.summary_plot.map(d => d.feature).reverse(), axisLabel: { color: textColor } },
                  series: [{
                    type: 'bar',
                    data: viewModel.shap.summary_plot.map(d => d.mean_shap).reverse(),
                    itemStyle: { color: '#FDA481', borderRadius: [0, 4, 4, 0] }
                  }]
                }}
                style={{ height: '100%', width: '100%' }}
              />
            ) : (
              <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-theme-border bg-theme-bg/40 text-sm text-theme-text-secondary">
                SHAP data unavailable
              </div>
            )}
          </div>
        </div>
      )}
      {activeTab === 'other-tools' && (
        <div className="glass-panel space-y-6">
          <div className="rounded-2xl border border-theme-primary/30 bg-theme-primary/5 p-5">
            <h3 className="text-sm font-bold text-theme-primary mb-2">🎓 What to look for: Local Explanations (LIME)</h3>
            <p className="text-sm text-theme-text-secondary">LIME approximates the model locally around a specific prediction. Look for the top weighted features to understand why a specific sample was classified the way it was. If a feature heavily swings a bad prediction, consider augmenting your dataset around that feature.</p>
          </div>
          <div>
            <h2 className="text-xl font-bold text-theme-text-primary">Local Explanations (LIME)</h2>
            <p className="mt-2 text-sm text-theme-text-secondary">Sample-level proxy explanations.</p>
          </div>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2 mt-4">
            {viewModel.lime?.samples?.map((sample) => (
              <div key={sample.index} className="glass-card">
                <div className="flex justify-between items-center mb-4">
                  <p className="font-semibold text-theme-text-primary">Sample #{sample.index}</p>
                  <span className="badge border-theme-primary/20 bg-theme-primary/10 text-theme-primary">
                    {sample.prediction} ({(sample.probability * 100).toFixed(1)}%)
                  </span>
                </div>
                <div className="space-y-2">
                  {sample.explanation?.map((expl, idx) => (
                    <div key={idx} className="flex justify-between text-sm items-center rounded-xl border border-theme-border/50 bg-theme-bg/50 px-3 py-2">
                      <span className="text-theme-text-secondary">{expl.feature}</span>
                      <span className={`font-mono ${expl.weight > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                        {expl.weight > 0 ? '+' : ''}{expl.weight.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
            {!viewModel.lime?.samples && (
              <div className="rounded-2xl border border-dashed border-theme-border bg-theme-bg/40 px-6 py-10 text-sm text-theme-text-secondary text-center col-span-2">
                No LIME explanations available yet.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
