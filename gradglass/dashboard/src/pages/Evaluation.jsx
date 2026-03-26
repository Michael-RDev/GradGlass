import React, { useEffect, useMemo, useState } from 'react'
import { useParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import { Activity, AlertTriangle, BrainCircuit, CheckCircle2, Layers3, Sparkles, Target } from 'lucide-react'

import { fetchEvalLab } from '../api'
import { EmptyState, ErrorMessage, LoadingSpinner } from '../components/ui'
import { useTheme } from '../components/ThemeProvider'

const PERCENT_METRICS = new Set([
  'accuracy',
  'precision',
  'recall',
  'macro_f1',
  'micro_f1',
  'mape',
  'bleu',
  'rouge_l',
  'semantic_similarity',
  'recall_at_1',
  'recall_at_5',
  'top_1_accuracy',
  'top_5_accuracy',
  'mean_iou',
  'mAP_50',
  'success_rate',
])

const TREND_LINE_COLORS = ['#B4182D', '#FDA481', '#37415C']

function formatMetricName(name) {
  if (!name) return 'Metric'
  const special = {
    macro_f1: 'Macro F1',
    micro_f1: 'Micro F1',
    mse: 'MSE',
    rmse: 'RMSE',
    mae: 'MAE',
    r2: 'R²',
    rouge_l: 'ROUGE-L',
    recall_at_1: 'Recall@1',
    recall_at_5: 'Recall@5',
    top_1_accuracy: 'Top-1 Accuracy',
    top_5_accuracy: 'Top-5 Accuracy',
    mean_iou: 'Mean IoU',
    mAP_50: 'mAP@0.50',
  }
  if (special[name]) return special[name]
  return name
    .split('_')
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(' ')
}

function formatMetricValue(name, value) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  const numeric = Number(value)
  if (PERCENT_METRICS.has(name)) {
    return `${(numeric <= 1 ? numeric * 100 : numeric).toFixed(2)}%`
  }
  if (Math.abs(numeric) >= 1000) {
    return numeric.toLocaleString(undefined, { maximumFractionDigits: 2 })
  }
  return numeric.toFixed(4)
}

function formatTaskName(name) {
  if (!name) return 'Unknown'
  const special = {
    retrieval_embedding: 'Retrieval / Embeddings',
    time_series_forecasting: 'Time Series Forecasting',
    sequence_generation: 'Sequence Generation',
    reinforcement_learning: 'Reinforcement Learning',
  }
  return special[name] || formatMetricName(name)
}

function formatBenchmarkFamily(name) {
  const special = {
    llm: 'LLM Benchmarks',
    vision: 'Vision Benchmarks',
  }
  return special[name] || formatMetricName(name)
}

function statusTone(status) {
  if (status === 'improving' || status === 'stable' || status === 'clear') {
    return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/20'
  }
  if (status === 'likely_significant') {
    return 'text-blue-500 bg-blue-500/10 border-blue-500/20'
  }
  if (status === 'moderate') {
    return 'text-amber-500 bg-amber-500/10 border-amber-500/20'
  }
  if (
    status === 'severe' ||
    status === 'degrading' ||
    status === 'detected' ||
    status === 'possible_overfitting' ||
    status === 'possible_underfitting'
  ) {
    return 'text-theme-secondary bg-theme-secondary/10 border-theme-secondary/20'
  }
  return 'text-theme-text-secondary bg-theme-bg border-theme-border'
}

function StatusPill({ status, label }) {
  return <span className={`badge ${statusTone(status)}`}>{label || (status || 'unknown').replaceAll('_', ' ')}</span>
}

function Section({ icon: Icon, title, children, action }) {
  return (
    <section className="card space-y-5">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 rounded-2xl bg-theme-primary/15 text-theme-primary flex items-center justify-center shadow-sm">
            <Icon className="w-5 h-5" />
          </div>
          <h2 className="h3 text-theme-text-primary">{title}</h2>
        </div>
        {action}
      </div>
      {children}
    </section>
  )
}

export default function Evaluation() {
  const { runId } = useParams()
  const [payload, setPayload] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const { theme } = useTheme()

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetchEvalLab(runId)
      .then((res) => setPayload(res))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [runId])

  const report = payload?.report || null
  const evaluations = report?.evaluations || payload?.evaluations || []
  const latestEvaluation = evaluations.length > 0 ? evaluations[evaluations.length - 1] : null

  const textColor = theme === 'dark' ? 'rgba(255,255,255,0.78)' : '#37415C'
  const gridColor = theme === 'dark' ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'

  const trendOptions = useMemo(() => {
    const trend = report?.trend_analysis
    if (!trend?.series?.length || !trend?.series_keys?.length) return null

    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { data: trend.series_keys.map(formatMetricName), textStyle: { color: textColor }, top: 0 },
      grid: { left: 44, right: 18, top: 34, bottom: 30 },
      xAxis: {
        type: 'category',
        data: trend.series.map((item, index) => item.step ?? index + 1),
        axisLabel: { color: textColor },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: textColor },
        splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
      },
      series: trend.series_keys.map((key, index) => ({
        name: formatMetricName(key),
        type: 'line',
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 3 },
        itemStyle: { color: TREND_LINE_COLORS[index % TREND_LINE_COLORS.length] },
        data: trend.series.map((item) => item[key]),
      })),
    }
  }, [gridColor, report?.trend_analysis, textColor])

  const confusionMatrixOptions = useMemo(() => {
    const matrixPayload = latestEvaluation?.confusion_matrix
    if (!matrixPayload?.matrix?.length || !matrixPayload?.classes?.length) return null

    const points = []
    matrixPayload.matrix.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        points.push([columnIndex, rowIndex, value])
      })
    })
    const maxValue = Math.max(...points.map((item) => item[2]), 0)

    return {
      tooltip: { position: 'top' },
      grid: { left: 70, right: 20, top: 10, bottom: 50 },
      xAxis: {
        type: 'category',
        data: matrixPayload.classes.map(String),
        name: 'Predicted',
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: { color: textColor },
        splitArea: { show: true },
      },
      yAxis: {
        type: 'category',
        data: matrixPayload.classes.map(String),
        name: 'Actual',
        nameLocation: 'middle',
        nameGap: 50,
        axisLabel: { color: textColor },
        splitArea: { show: true },
      },
      visualMap: {
        min: 0,
        max: maxValue,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        textStyle: { color: textColor },
        inRange: {
          color: theme === 'dark' ? ['#242E49', '#FDA481'] : ['#F0F2F7', '#FDA481'],
        },
      },
      series: [
        {
          name: 'Confusion Matrix',
          type: 'heatmap',
          data: points,
          label: { show: true, color: '#181A2F' },
          emphasis: { itemStyle: { shadowBlur: 12, shadowColor: 'rgba(0,0,0,0.25)' } },
        },
      ],
    }
  }, [latestEvaluation?.confusion_matrix, textColor, theme])

  if (loading) return <LoadingSpinner />
  if (error) return <ErrorMessage message={error} />

  if (!report && evaluations.length === 0) {
    return (
      <EmptyState
        icon={Activity}
        title="No evaluation data found"
        description="Log predictions with `run.log_batch()` or metrics with `run.log()` to generate a task-aware evaluation report."
      />
    )
  }

  const distributionEntries = Object.entries(report?.task_type_distribution || {})
  const selectedMetrics = report?.selected_metrics || []
  const headlineMetrics = report?.performance_summary?.headline_metrics || []
  const benchmarkState = report?.benchmark_state || { enabled: false, eligible_families: [], message: null }
  const eligibleBenchmarkFamilies = benchmarkState?.eligible_families || []
  const weakClasses = report?.error_analysis?.weak_classes || []
  const dominantMistakes = report?.error_analysis?.dominant_misclassifications || []
  const missingArtifacts = report?.missing_artifacts || []
  const recommendations = report?.recommendations || []
  const summaryLines = report?.error_analysis?.summary || []
  const modalities = report?.modality_analysis?.modalities || []
  const crossModal = report?.modality_analysis?.cross_modal_alignment || null
  const significance = report?.trend_analysis?.significance || null
  const overconfident = report?.error_analysis?.overconfident_errors || null
  const imbalance = report?.error_analysis?.data_imbalance || null
  const generalization = report?.error_analysis?.generalization || null

  return (
    <div className="space-y-6 pb-10">
      <Section
        icon={BrainCircuit}
        title="1. Inferred Task Type"
        action={
          <StatusPill
            status={report?.confidence_in_task_inference >= 0.7 ? 'clear' : 'moderate'}
            label={`${((report?.confidence_in_task_inference || 0) * 100).toFixed(1)}% confidence`}
          />
        }
      >
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-5">
          <div className="xl:col-span-5 rounded-3xl border border-theme-border bg-theme-bg/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <Target className="w-5 h-5 text-theme-primary" />
              <span className="text-xs uppercase tracking-[0.24em] text-theme-text-muted font-semibold">Current Best Inference</span>
            </div>
            <div>
              <div className="text-4xl font-semibold text-theme-text-primary leading-tight">{report?.inferred_task_type_display}</div>
              <p className="text-sm text-theme-text-secondary mt-2">{report?.performance_summary?.summary}</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {modalities.map((modality) => (
                <span key={modality} className="badge text-theme-primary bg-theme-primary/10 border-theme-primary/20">
                  {modality.replaceAll('_', ' ')}
                </span>
              ))}
            </div>
          </div>

          <div className="xl:col-span-7 rounded-3xl border border-theme-border bg-theme-surface/60 p-6 space-y-4">
            <div className="flex items-center gap-3">
              <Layers3 className="w-5 h-5 text-theme-secondary" />
              <span className="text-xs uppercase tracking-[0.24em] text-theme-text-muted font-semibold">Task Distribution</span>
            </div>
            <div className="space-y-3">
              {distributionEntries.map(([task, probability]) => (
                <div key={task} className="space-y-1.5">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-theme-text-primary font-medium">{formatTaskName(task)}</span>
                    <span className="text-theme-text-secondary font-mono">{(probability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full h-2.5 rounded-full bg-theme-bg overflow-hidden">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-theme-primary to-theme-secondary"
                      style={{ width: `${Math.max(probability * 100, 3)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
            {report?.task_inference_evidence?.length > 0 && (
              <div className="rounded-2xl border border-theme-border bg-theme-bg/50 p-4">
                <p className="text-sm font-medium text-theme-text-primary mb-2">Inference evidence</p>
                <ul className="space-y-2 text-sm text-theme-text-secondary">
                  {report.task_inference_evidence.map((item, index) => (
                    <li key={index}>• {item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </Section>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-7 space-y-6">
          <Section icon={Sparkles} title="2. Selected Metrics & Justification">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {selectedMetrics.map((metric) => (
                <div key={metric.name} className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4 space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <h3 className="font-semibold text-theme-text-primary">{metric.display_name || formatMetricName(metric.name)}</h3>
                    <span className="text-theme-primary text-xs font-mono">{metric.name}</span>
                  </div>
                  <p className="text-sm text-theme-text-secondary">{metric.justification}</p>
                </div>
              ))}
            </div>
          </Section>

          <Section icon={Activity} title="3. Performance Summary">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {headlineMetrics.map((metric) => (
                <div key={metric.name} className="rounded-[24px] border border-theme-border bg-gradient-to-br from-theme-surface to-theme-bg p-5">
                  <p className="text-sm text-theme-text-secondary font-medium">{metric.label}</p>
                  <div className="text-3xl font-bold text-theme-text-primary mt-2">{metric.display}</div>
                </div>
              ))}
            </div>
            <div className="rounded-2xl border border-theme-border bg-theme-bg/60 p-4 space-y-2">
              <p className="text-sm font-medium text-theme-text-primary">Summary</p>
              <p className="text-sm text-theme-text-secondary">{report?.performance_summary?.summary}</p>
              {benchmarkState?.enabled && (
                <div className="pt-3 border-t border-theme-border/60 space-y-3">
                  <div className="space-y-2">
                    <p className="text-xs font-medium uppercase tracking-[0.18em] text-theme-text-muted">Benchmark Suites</p>
                    {eligibleBenchmarkFamilies.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        {eligibleBenchmarkFamilies.map((family) => (
                          <span key={family} className="badge text-theme-primary bg-theme-primary/10 border-theme-primary/20">
                            {formatBenchmarkFamily(family)}
                          </span>
                        ))}
                      </div>
                    )}
                    {benchmarkState?.message && (
                      <p className="text-xs text-theme-text-muted">{benchmarkState.message}</p>
                    )}
                  </div>
                  {eligibleBenchmarkFamilies.length > 0 && report?.benchmark_alignment?.message && (
                    <p className="text-xs text-theme-text-muted">{report.benchmark_alignment.message}</p>
                  )}
                </div>
              )}
            </div>
          </Section>

          <Section icon={Activity} title="4. Trend Analysis" action={<StatusPill status={report?.trend_analysis?.status} />}>
            <div className="space-y-4">
              <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                <p className="text-sm text-theme-text-secondary">{report?.trend_analysis?.summary}</p>
              </div>
              {trendOptions ? (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 h-[360px]">
                  <ReactECharts option={trendOptions} style={{ height: '100%', width: '100%' }} />
                </div>
              ) : (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 text-sm text-theme-text-secondary">
                  Trend analysis needs at least two evaluation snapshots.
                </div>
              )}
              {significance && (
                <div className="flex items-start gap-3 rounded-2xl border border-theme-border bg-theme-bg/50 p-4">
                  <StatusPill status={significance.status} />
                  <p className="text-sm text-theme-text-secondary">{significance.message}</p>
                </div>
              )}
            </div>
          </Section>
        </div>

        <div className="xl:col-span-5 space-y-6">
          <Section icon={AlertTriangle} title="5. Error Analysis">
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-3">
                {generalization && (
                  <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="font-medium text-theme-text-primary">Generalization</p>
                      <StatusPill status={generalization.status} />
                    </div>
                    <p className="text-sm text-theme-text-secondary mt-2">{generalization.message}</p>
                  </div>
                )}
                {imbalance && (
                  <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="font-medium text-theme-text-primary">Data Imbalance</p>
                      <StatusPill status={imbalance.status} />
                    </div>
                    <p className="text-sm text-theme-text-secondary mt-2">{imbalance.message}</p>
                  </div>
                )}
                {overconfident && (
                  <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                    <div className="flex items-center justify-between gap-3">
                      <p className="font-medium text-theme-text-primary">Overconfident Errors</p>
                      <StatusPill status={overconfident.status} />
                    </div>
                    <p className="text-sm text-theme-text-secondary mt-2">{overconfident.message}</p>
                  </div>
                )}
              </div>

              {summaryLines.length > 0 && (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                  <p className="font-medium text-theme-text-primary mb-2">Key findings</p>
                  <ul className="space-y-2 text-sm text-theme-text-secondary">
                    {summaryLines.map((line, index) => (
                      <li key={index}>• {line}</li>
                    ))}
                  </ul>
                </div>
              )}

              {weakClasses.length > 0 && (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                  <p className="font-medium text-theme-text-primary mb-3">Weak Classes / Labels</p>
                  <div className="space-y-3">
                    {weakClasses.slice(0, 5).map((item, index) => (
                      <div key={`${item.class}-${index}`} className="rounded-xl border border-theme-border bg-theme-surface/70 p-3">
                        <div className="flex items-center justify-between gap-3">
                          <span className="font-medium text-theme-text-primary">{String(item.class)}</span>
                          <span className="text-xs text-theme-text-muted font-mono">n={item.support}</span>
                        </div>
                        <div className="grid grid-cols-3 gap-3 text-sm mt-3">
                          <div>
                            <div className="text-theme-text-muted">Precision</div>
                            <div className="font-mono text-theme-text-primary">{formatMetricValue('precision', item.precision)}</div>
                          </div>
                          <div>
                            <div className="text-theme-text-muted">Recall</div>
                            <div className="font-mono text-theme-text-primary">{formatMetricValue('recall', item.recall)}</div>
                          </div>
                          <div>
                            <div className="text-theme-text-muted">F1</div>
                            <div className="font-mono text-theme-text-primary">{formatMetricValue('macro_f1', item.f1)}</div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {dominantMistakes.length > 0 && (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                  <p className="font-medium text-theme-text-primary mb-3">Dominant Misclassification Patterns</p>
                  <div className="space-y-2">
                    {dominantMistakes.slice(0, 5).map((item, index) => (
                      <div key={`${item.actual}-${item.predicted}-${index}`} className="flex items-center justify-between gap-3 text-sm text-theme-text-secondary">
                        <span>{String(item.actual)} → {String(item.predicted)}</span>
                        <span className="font-mono text-theme-text-primary">{item.count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Section>

          {crossModal && modalities.length > 1 && (
            <Section icon={Layers3} title="Cross-Modal Alignment">
              <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4">
                <div className="flex items-center justify-between gap-3">
                  <p className="font-medium text-theme-text-primary">Cross-modal check</p>
                  <StatusPill status={crossModal.status} />
                </div>
                <p className="text-sm text-theme-text-secondary mt-2">{crossModal.message}</p>
              </div>
            </Section>
          )}
        </div>
      </div>

      {confusionMatrixOptions && (
        <Section icon={CheckCircle2} title="Confusion Matrix">
          <div className="rounded-2xl border border-theme-border bg-theme-bg/40 p-4 h-[440px]">
            <ReactECharts option={confusionMatrixOptions} style={{ height: '100%', width: '100%' }} />
          </div>
        </Section>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
        <div className="xl:col-span-7">
          <Section icon={Sparkles} title="6. Recommendations">
            <div className="space-y-3">
              {recommendations.length > 0 ? (
                recommendations.map((recommendation, index) => (
                  <div key={index} className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4 text-sm text-theme-text-secondary">
                    <span className="font-semibold text-theme-text-primary mr-2">{index + 1}.</span>
                    {recommendation}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4 text-sm text-theme-text-secondary">
                  No extra action is suggested from the currently logged evidence.
                </div>
              )}
            </div>
          </Section>
        </div>

        <div className="xl:col-span-5">
          <Section icon={AlertTriangle} title="7. Missing Artifacts">
            <div className="space-y-3">
              {missingArtifacts.length > 0 ? (
                missingArtifacts.map((item, index) => (
                  <div key={index} className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4 text-sm text-theme-text-secondary">
                    {item}
                  </div>
                ))
              ) : (
                <div className="rounded-2xl border border-theme-border bg-theme-bg/55 p-4 text-sm text-theme-text-secondary">
                  No critical evaluation artifacts appear to be missing for the current task inference.
                </div>
              )}
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}
