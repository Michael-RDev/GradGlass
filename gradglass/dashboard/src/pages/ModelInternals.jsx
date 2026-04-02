import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import {
  BarChart2,
  Box,
  Check,
  Copy,
  Image as ImageIcon,
  Layers,
  Network,
  Share2,
  Snowflake,
  Sparkles,
  Zap,
  ZoomIn,
} from 'lucide-react';
import {
  fetchDistributions,
  fetchEmbeddings,
  fetchFreezeCode,
  fetchGradients,
  fetchSaliency,
} from '../api';
import { useTheme } from '../components/ThemeProvider';
import useRunStore from '../store/useRunStore';
import ArchitectureGraph from './ArchitectureGraph';

const TABS = [
  { id: 'architecture', label: 'Architecture & Gradients', icon: Network },
  { id: 'ablation', label: 'Layer Ablation', icon: Snowflake },
  { id: 'distributions', label: 'Distributions', icon: BarChart2 },
  { id: 'saliency', label: 'Saliency Maps', icon: ImageIcon },
  { id: 'embeddings', label: 'Embeddings', icon: Box },
];

const GRADIENT_STATUS_META = {
  too_small: { label: 'Too small', color: '#2563eb', chip: 'bg-blue-500/10 text-blue-400 border-blue-500/20' },
  healthy: { label: 'Healthy range', color: '#f97316', chip: 'bg-orange-500/10 text-orange-400 border-orange-500/20' },
  too_large: { label: 'Too large', color: '#dc2626', chip: 'bg-red-500/10 text-red-400 border-red-500/20' },
  noisy: { label: 'Noisy', color: '#eab308', chip: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' },
};

const EMBEDDING_COLORS = ['#6366f1', '#f97316', '#10b981', '#ec4899', '#38bdf8', '#facc15', '#ef4444', '#8b5cf6'];

function normalizeLayerId(layerId) {
  return (layerId || '').replace(/\.(weight|bias|running_mean|running_var|num_batches_tracked|gamma|beta)$/i, '');
}

function normalizeStatus(status) {
  const value = (status || '').toString().trim().toLowerCase();
  if (value === 'complete') return 'completed';
  if (value === 'finished') return 'completed';
  return value;
}

function isTerminalStatus(status) {
  return ['completed', 'failed', 'cancelled', 'interrupted'].includes(normalizeStatus(status));
}

function formatNumber(value, digits = 4) {
  if (value == null || Number.isNaN(value)) return '—';
  return Number(value).toFixed(digits);
}

function buildHistogramOption(entry, textColor, gridColor) {
  if (!entry?.histogram?.counts?.length || !entry?.histogram?.bin_edges?.length) {
    return null;
  }

  const binEdges = entry.histogram.bin_edges;
  const labels = entry.histogram.counts.map((_, index) => {
    const left = Number(binEdges[index] || 0);
    const right = Number(binEdges[index + 1] || 0);
    const midpoint = (left + right) / 2;
    return midpoint.toFixed(3);
  });

  return {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 56, right: 24, top: 20, bottom: 44 },
    xAxis: {
      type: 'category',
      data: labels,
      axisLabel: { color: textColor, fontSize: 10, hideOverlap: true },
      axisLine: { lineStyle: { color: gridColor } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: textColor, fontSize: 10 },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    series: [
      {
        type: 'bar',
        data: entry.histogram.counts,
        barWidth: '88%',
        itemStyle: {
          color: entry.kind === 'activations' ? '#14b8a6' : '#6366f1',
          borderRadius: [4, 4, 0, 0],
        },
      },
    ],
  };
}

function buildStructuredSaliencyOption(featureImportance, textColor, gridColor) {
  const topFeatures = [...(featureImportance || [])].slice(0, 12).reverse();
  return {
    animation: false,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    grid: { left: 120, right: 24, top: 20, bottom: 24 },
    xAxis: {
      type: 'value',
      axisLabel: { color: textColor },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    yAxis: {
      type: 'category',
      data: topFeatures.map((feature) => feature.feature),
      axisLabel: { color: textColor, fontSize: 11 },
    },
    series: [
      {
        type: 'bar',
        data: topFeatures.map((feature) => feature.score),
        itemStyle: { color: '#ec4899', borderRadius: [0, 4, 4, 0] },
      },
    ],
  };
}

function colorForLabel(label, fallbackIndex = 0) {
  if (label == null) return '#94a3b8';
  const value = Number.isFinite(Number(label)) ? Number(label) : fallbackIndex;
  const normalized = Math.abs(Math.round(value));
  return EMBEDDING_COLORS[normalized % EMBEDDING_COLORS.length];
}

function buildEmbeddingOption(layer, payload, colorMode, textColor, gridColor) {
  if (!layer?.projection?.length) {
    return null;
  }

  const labels = colorMode === 'prediction' ? payload?.predictions : payload?.targets;
  const fallbackLabel = colorMode === 'prediction' ? payload?.targets : payload?.predictions;
  const confidence = payload?.confidence || [];
  const data = layer.projection.map((point, index) => {
    const label = labels?.[index] ?? fallbackLabel?.[index] ?? index;
    return {
      value: point,
      itemStyle: { color: colorForLabel(label, index) },
      meta: {
        index,
        target: payload?.targets?.[index] ?? null,
        prediction: payload?.predictions?.[index] ?? null,
        confidence: confidence?.[index] ?? null,
      },
    };
  });

  return {
    animation: false,
    tooltip: {
      trigger: 'item',
      formatter: (params) => {
        const meta = params?.data?.meta || {};
        const lines = [
          `Sample ${meta.index ?? '—'}`,
          `Target: ${meta.target ?? '—'}`,
          `Prediction: ${meta.prediction ?? '—'}`,
        ];
        if (meta.confidence != null) {
          lines.push(`Confidence: ${(Number(meta.confidence) * 100).toFixed(1)}%`);
        }
        return lines.join('<br/>');
      },
    },
    grid: { left: 36, right: 20, top: 20, bottom: 32 },
    xAxis: {
      type: 'value',
      name: 'PC 1',
      nameTextStyle: { color: textColor },
      axisLabel: { color: textColor },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    yAxis: {
      type: 'value',
      name: 'PC 2',
      nameTextStyle: { color: textColor },
      axisLabel: { color: textColor },
      splitLine: { lineStyle: { color: gridColor, type: 'dashed' } },
    },
    series: [
      {
        type: 'scatter',
        data,
        symbolSize: 11,
      },
    ],
  };
}

function HeatmapGrid({ values, tone = 'input' }) {
  const rows = Array.isArray(values) ? values : [];
  const cols = rows[0]?.length || 0;
  const baseColor =
    tone === 'saliency'
      ? (value) => `rgba(244, 63, 94, ${Math.min(1, Math.max(0.08, value))})`
      : (value) => `rgba(148, 163, 184, ${Math.min(1, Math.max(0.06, value))})`;

  return (
    <div
      className="grid aspect-square w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-950/70 dark:border-slate-800"
      style={{ gridTemplateColumns: `repeat(${cols || 1}, minmax(0, 1fr))` }}
    >
      {rows.flatMap((row, rowIndex) =>
        row.map((value, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            style={{ background: baseColor(Number(value) || 0) }}
            className="aspect-square"
          />
        ))
      )}
    </div>
  );
}

function CopyButton({ label, text, activeCopyId, setActiveCopyId, copyId }) {
  const copied = activeCopyId === copyId;

  async function onCopy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setActiveCopyId(copyId);
      window.setTimeout(() => setActiveCopyId((current) => (current === copyId ? null : current)), 1800);
    } catch {
      setActiveCopyId(null);
    }
  }

  return (
    <button
      onClick={onCopy}
      className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-200 transition hover:border-slate-600 hover:bg-slate-800"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-emerald-400" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : label}
    </button>
  );
}

export default function ModelInternals() {
  const { runId } = useParams();
  const { theme } = useTheme();
  const { metadata, overview } = useRunStore();
  const [gradients, setGradients] = useState(null);
  const [freezeCode, setFreezeCode] = useState(null);
  const [distributions, setDistributions] = useState(null);
  const [saliency, setSaliency] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('architecture');
  const [selectedDagNode, setSelectedDagNode] = useState(null);
  const [selectedArchitectureNodeId, setSelectedArchitectureNodeId] = useState(null);
  const [distributionMode, setDistributionMode] = useState('activations');
  const [distributionLayer, setDistributionLayer] = useState(null);
  const [embeddingLayer, setEmbeddingLayer] = useState(null);
  const [embeddingColorMode, setEmbeddingColorMode] = useState('target');
  const [activeCopyId, setActiveCopyId] = useState(null);

  useEffect(() => {
    if (!runId) return;

    setLoading(true);
    Promise.all([
      fetchGradients(runId).catch(() => null),
      fetchFreezeCode(runId).catch(() => null),
      fetchDistributions(runId).catch(() => null),
      fetchSaliency(runId).catch(() => null),
      fetchEmbeddings(runId).catch(() => null),
    ])
      .then(([gradientPayload, freezePayload, distributionPayload, saliencyPayload, embeddingPayload]) => {
        setGradients(gradientPayload);
        setFreezeCode(freezePayload);
        setDistributions(distributionPayload);
        setSaliency(saliencyPayload);
        setEmbeddings(embeddingPayload);
      })
      .finally(() => setLoading(false));
  }, [runId]);

  const textColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.75)' : '#37415C';
  const gridColor = theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.06)';
  const resolvedStatus = normalizeStatus(overview?.status || metadata?.status);
  const terminalRun = isTerminalStatus(resolvedStatus);

  const gradientOptions = useMemo(() => {
    if (!gradients?.summaries?.length) {
      return { latestStep: 0, numLayers: 0, option: null };
    }

    const latest = gradients.summaries[gradients.summaries.length - 1];
    const analysisByLayer = new Map((gradients.analysis || []).map((entry) => [entry.layer, entry]));
    const layers = Object.keys(latest.layers || {}).sort().reverse();
    const seriesData = layers.map((layer) => {
      const norm = Number(latest.layers[layer]?.norm || 0);
      const status = analysisByLayer.get(layer)?.stability_status || 'healthy';
      const meta = GRADIENT_STATUS_META[status] || GRADIENT_STATUS_META.healthy;
      return {
        value: Math.max(norm, 1e-12),
        rawValue: norm,
        layer,
        status,
        reason: analysisByLayer.get(layer)?.stability_reason || 'Gradient norm is in a healthy range.',
        itemStyle: { color: meta.color },
      };
    });

    return {
      latestStep: latest.step,
      numLayers: layers.length,
      option: {
        animation: false,
        tooltip: {
          trigger: 'item',
          formatter: (params) => {
            const data = params?.data || {};
            const meta = GRADIENT_STATUS_META[data.status] || GRADIENT_STATUS_META.healthy;
            return [
              `<strong>${data.layer}</strong>`,
              `L2 norm: ${Number(data.rawValue || 0).toExponential(3)}`,
              `Status: ${meta.label}`,
              data.reason,
            ].join('<br/>');
          },
        },
        grid: { left: 180, right: 24, bottom: 32, top: 20 },
        xAxis: {
          type: 'log',
          name: 'Gradient L2 norm',
          nameTextStyle: { color: textColor },
          axisLabel: { color: textColor },
          splitLine: { lineStyle: { color: gridColor } },
        },
        yAxis: {
          type: 'category',
          data: layers,
          axisLabel: { color: textColor, fontSize: 10, width: 160, overflow: 'truncate' },
        },
        series: [
          {
            type: 'bar',
            data: seriesData,
          },
        ],
      },
    };
  }, [gradients, gridColor, textColor]);

  const distributionCollections = useMemo(
    () => ({
      activations: distributions?.activations?.layers || [],
      weights: distributions?.weights?.layers || [],
    }),
    [distributions]
  );

  useEffect(() => {
    const requestedMode = distributionCollections[distributionMode]?.length ? distributionMode : null;
    const fallbackMode = distributionCollections.activations.length
      ? 'activations'
      : distributionCollections.weights.length
        ? 'weights'
        : null;
    const nextMode = requestedMode || fallbackMode;

    if (nextMode && nextMode !== distributionMode) {
      setDistributionMode(nextMode);
      return;
    }

    const layerOptions = nextMode ? distributionCollections[nextMode] : [];
    if (!layerOptions.length) {
      setDistributionLayer(null);
      return;
    }

    if (!distributionLayer || !layerOptions.some((entry) => entry.layer === distributionLayer)) {
      setDistributionLayer(layerOptions[0].layer);
    }
  }, [distributionCollections, distributionLayer, distributionMode]);

  const selectedDistribution = useMemo(() => {
    const entries = distributionCollections[distributionMode] || [];
    return entries.find((entry) => entry.layer === distributionLayer) || entries[0] || null;
  }, [distributionCollections, distributionLayer, distributionMode]);

  useEffect(() => {
    if (!embeddings?.available || !embeddings?.layers?.length) {
      setEmbeddingLayer(null);
      return;
    }
    if (!embeddingLayer || !embeddings.layers.some((layer) => layer.layer === embeddingLayer)) {
      setEmbeddingLayer(embeddings.default_layer || embeddings.layers[0].layer);
    }
  }, [embeddingLayer, embeddings]);

  const selectedEmbedding = useMemo(() => {
    if (!embeddings?.layers?.length) return null;
    return embeddings.layers.find((layer) => layer.layer === embeddingLayer) || embeddings.layers[0];
  }, [embeddingLayer, embeddings]);

  const selectedPatch = useMemo(() => {
    if (!freezeCode) return '';
    return metadata?.framework === 'tensorflow' ? freezeCode.tensorflow_code || freezeCode.pytorch_code : freezeCode.pytorch_code || freezeCode.tensorflow_code;
  }, [freezeCode, metadata?.framework]);

  if (loading) {
    return <div className="p-8 text-slate-500 dark:text-slate-400">Loading visualization data...</div>;
  }

  return (
    <div className="space-y-6 flex flex-col h-full min-h-[calc(100vh-8rem)]">
      <div className="flex items-center justify-between shrink-0">
        <div>
          <h1 className="h2 text-theme-text-primary">Visualizations Hub</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Inspect architecture, gradients, activations, saliency, and representation structure from the latest captured snapshot.
          </p>
        </div>
        <div className="flex gap-4">
          <button className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg text-sm font-medium text-slate-700 dark:text-slate-300 shadow-sm hover:bg-slate-50 dark:hover:bg-slate-800">
            <Share2 className="w-4 h-4" /> Export Report
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 shrink-0">
        <div className="card flex flex-col justify-between hover:border-theme-primary transition-colors cursor-default">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Tracked Gradients</p>
            <Layers className="w-4 h-4 text-theme-primary" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{gradientOptions.numLayers?.toLocaleString() || '--'}</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Latest Capture</p>
            <Zap className="w-4 h-4 text-emerald-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{gradientOptions.latestStep?.toLocaleString() || '--'}</p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Probe Layers</p>
            <ZoomIn className="w-4 h-4 text-violet-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">
            {(embeddings?.layers?.length || distributions?.activations?.layers?.length || 0).toLocaleString()}
          </p>
        </div>

        <div className="card flex flex-col justify-between">
          <div className="flex items-center justify-between mb-2 opacity-80">
            <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Ablation Candidates</p>
            <Snowflake className="w-4 h-4 text-blue-500" />
          </div>
          <p className="text-3xl font-bold text-slate-900 dark:text-white">{freezeCode?.candidates?.length || 0}</p>
        </div>
      </div>

      {terminalRun && (
        <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100 shadow-sm">
          You are viewing the final captured model snapshot for this completed run. The architecture view stays interactive so you can inspect layers and expand groups after training ends.
        </div>
      )}

      <div className="flex gap-2 border-b border-theme-border pb-px shrink-0 overflow-x-auto custom-scrollbar">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === tab.id
                ? 'border-theme-primary text-theme-primary'
                : 'border-transparent text-theme-text-secondary hover:text-theme-text-primary hover:border-theme-border'
            }`}
          >
            <tab.icon className={`w-4 h-4 ${activeTab === tab.id ? 'text-theme-primary' : 'opacity-70'}`} />
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex-1 min-h-0 flex flex-col gap-6">
        {activeTab === 'architecture' && (
          <>
            <div className="card p-0 overflow-hidden flex flex-col relative h-[60vh] min-h-[520px] border-theme-border shadow-md">
              <ArchitectureGraph
                hideHeader
                selectedNodeId={selectedArchitectureNodeId}
                onNodeSelect={(node) => {
                  setSelectedDagNode(node);
                  setSelectedArchitectureNodeId(node?.id || null);
                }}
                heightClass="h-full"
              />
              {selectedDagNode && (
                <div className="absolute top-4 left-4 bg-slate-900/90 backdrop-blur border border-indigo-500/30 p-3 rounded-lg shadow-xl z-10 pointer-events-none fade-in">
                  <p className="text-xs text-indigo-300 font-mono mb-1">Selected Layer</p>
                  <p className="text-sm text-white font-bold">{selectedDagNode.id}</p>
                  {selectedDagNode.param_count > 0 && (
                    <p className="text-xs text-slate-400 mt-1">{selectedDagNode.param_count.toLocaleString()} params</p>
                  )}
                </div>
              )}
            </div>

            {!gradientOptions.option ? (
              <div className="card p-8 text-slate-500 dark:text-slate-400">
                No gradient data found. Run with <code className="text-pink-400">gradients='summary'</code>.
              </div>
            ) : (
              <div className="card shadow-md border-theme-border">
                <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
                  <h3 className="h3 text-theme-text-primary">Global Gradient Stability</h3>
                  <span className="text-xs px-2 py-0.5 rounded-md bg-indigo-100 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400">
                    Captured at step {gradientOptions.latestStep}
                  </span>
                </div>
                <p className="text-sm text-slate-500 dark:text-slate-400 mb-4">
                  How strong each layer&apos;s gradients are right now. Very small bars can mean the layer is learning too slowly; very large bars can mean updates are unstable.
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {Object.entries(GRADIENT_STATUS_META).map(([key, meta]) => (
                    <span key={key} className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs ${meta.chip}`}>
                      <span className="h-2 w-2 rounded-full" style={{ background: meta.color }} />
                      {meta.label}
                    </span>
                  ))}
                </div>
                <div className="h-[400px] overflow-y-auto pr-2 custom-scrollbar border border-slate-100 dark:border-slate-800/50 rounded-lg">
                  <ReactECharts option={gradientOptions.option} style={{ height: `${Math.max(400, gradientOptions.numLayers * 22)}px`, width: '100%' }} />
                </div>
              </div>
            )}
          </>
        )}

        {activeTab === 'ablation' && (
          <div className="card shadow-md border-theme-border">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
              <div>
                <h3 className="h3 text-theme-text-primary flex items-center gap-2">
                  <Snowflake className="w-5 h-5 text-blue-500" />
                  Layer Ablation / Freeze Tools
                </h3>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-2 max-w-3xl">
                  {freezeCode?.message || 'GradGlass ranks layers by relative gradient activity so you can spot likely freeze candidates without guessing.'}
                </p>
              </div>
              {selectedPatch && (
                <div className="flex flex-wrap gap-2">
                  <CopyButton
                    label="Copy Primary Patch"
                    text={selectedPatch}
                    activeCopyId={activeCopyId}
                    setActiveCopyId={setActiveCopyId}
                    copyId="primary-patch"
                  />
                  {freezeCode?.tensorflow_code && metadata?.framework !== 'tensorflow' && (
                    <CopyButton
                      label="Copy TensorFlow Patch"
                      text={freezeCode.tensorflow_code}
                      activeCopyId={activeCopyId}
                      setActiveCopyId={setActiveCopyId}
                      copyId="tf-patch"
                    />
                  )}
                </div>
              )}
            </div>

            {freezeCode?.candidates?.length ? (
              <div className="grid gap-3 grid-cols-1 md:grid-cols-2 xl:grid-cols-3 mb-6">
                {freezeCode.candidates.map((candidate) => {
                  const normalizedLayer = normalizeLayerId(candidate.layer);
                  const meanNorm = candidate.mean_grad_norm ?? 0;
                  const relativePct = (candidate.relative_norm || 0) * 100;
                  return (
                    <button
                      key={candidate.layer}
                      onClick={() => {
                        setSelectedArchitectureNodeId(normalizedLayer);
                        setActiveTab('architecture');
                      }}
                      className="text-left p-4 bg-slate-50 dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 shadow-sm hover:shadow-md hover:border-indigo-500/40 transition-all"
                    >
                      <div className="flex items-start justify-between gap-3 mb-3">
                        <p className="text-sm font-mono text-slate-800 dark:text-slate-100 truncate" title={candidate.layer}>
                          {normalizedLayer}
                        </p>
                        <span className="inline-flex items-center gap-1 rounded-full border border-blue-500/20 bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-400">
                          Inspect
                        </span>
                      </div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-slate-500">Relative activity</span>
                        <span className="text-xs font-mono font-medium text-blue-600 dark:text-blue-400">{relativePct.toFixed(4)}%</span>
                      </div>
                      <div className="w-full bg-slate-200 dark:bg-slate-800 h-1.5 rounded-full overflow-hidden mb-3">
                        <div className="bg-blue-500 h-full rounded-full transition-all" style={{ width: `${Math.max(2, Math.min(100, relativePct))}%` }} />
                      </div>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Mean L2 norm: <span className="font-mono text-slate-300">{meanNorm.toExponential(3)}</span>
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">
                        Stayed below 1% of the strongest observed layer across captured summaries, so it is a good first candidate for freezing or ablation.
                      </p>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="p-8 border border-slate-200 dark:border-slate-800 border-dashed rounded-xl bg-slate-50 dark:bg-slate-900/30 flex flex-col items-center justify-center text-slate-500 mb-6">
                <Snowflake className="w-8 h-8 text-slate-400 mb-3 opacity-50" />
                <p className="text-sm text-center">No layers are inactive enough to freeze safely across recent steps.</p>
              </div>
            )}

            {selectedPatch && (
              <div className="mt-4">
                <p className="text-xs font-semibold text-slate-500 dark:text-slate-500 uppercase tracking-wider mb-2">Generated Patch</p>
                <div className="bg-[#0f111a] p-5 rounded-xl overflow-x-auto text-sm font-mono text-slate-300 custom-scrollbar border border-slate-800/80 shadow-inner">
                  <pre className="leading-relaxed">{selectedPatch}</pre>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'distributions' && (
          <div className="card shadow-md border-theme-border flex flex-col gap-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="h3 text-theme-text-primary">Weight & Activation Distributions</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                  Inspect the latest checkpoint weights and the latest probe activations to spot dead, saturated, or skewed layers quickly.
                </p>
              </div>
              <div className="flex gap-2">
                {['activations', 'weights'].map((mode) => (
                  <button
                    key={mode}
                    disabled={!distributionCollections[mode]?.length}
                    onClick={() => setDistributionMode(mode)}
                    className={`rounded-lg px-3 py-2 text-sm border transition ${
                      distributionMode === mode
                        ? 'border-theme-primary bg-theme-primary/10 text-theme-primary'
                        : 'border-slate-200 dark:border-slate-800 text-theme-text-secondary disabled:opacity-40'
                    }`}
                  >
                    {mode === 'activations' ? 'Activations' : 'Weights'}
                  </button>
                ))}
              </div>
            </div>

            {!selectedDistribution ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-6 py-10 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/30 dark:text-slate-400">
                {distributions?.activations?.reason || distributions?.weights?.reason || 'No distribution data is available for this run yet.'}
              </div>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-3">
                  <select
                    value={selectedDistribution.layer}
                    onChange={(event) => setDistributionLayer(event.target.value)}
                    className="min-w-[240px] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
                  >
                    {(distributionCollections[distributionMode] || []).map((entry) => (
                      <option key={entry.layer} value={entry.layer}>
                        {entry.layer}
                      </option>
                    ))}
                  </select>
                  <span className="rounded-full border border-indigo-500/20 bg-indigo-500/10 px-2.5 py-1 text-xs text-indigo-400">
                    {distributionMode === 'activations'
                      ? `Probe step ${distributions?.activations?.step ?? '—'}`
                      : `Checkpoint step ${distributions?.weights?.step ?? '—'}`}
                  </span>
                  {selectedDistribution.warnings?.map((warning) => (
                    <span key={warning} className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2.5 py-1 text-xs text-amber-400 capitalize">
                      {warning}
                    </span>
                  ))}
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="card">
                    <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Mean</p>
                    <p className="text-lg font-mono text-theme-text-primary">{formatNumber(selectedDistribution.stats?.mean, 5)}</p>
                  </div>
                  <div className="card">
                    <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Std Dev</p>
                    <p className="text-lg font-mono text-theme-text-primary">{formatNumber(selectedDistribution.stats?.std, 5)}</p>
                  </div>
                  <div className="card">
                    <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Sparsity</p>
                    <p className="text-lg font-mono text-theme-text-primary">{((selectedDistribution.stats?.sparsity || 0) * 100).toFixed(2)}%</p>
                  </div>
                  <div className="card">
                    <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Range</p>
                    <p className="text-lg font-mono text-theme-text-primary">
                      {formatNumber(selectedDistribution.stats?.min, 3)} to {formatNumber(selectedDistribution.stats?.max, 3)}
                    </p>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-950/60">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-theme-text-primary">{selectedDistribution.layer}</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        {selectedDistribution.kind === 'activations' ? 'Latest probe activation histogram' : 'Latest checkpoint weight histogram'}
                      </p>
                    </div>
                    <span className="text-xs text-slate-500 dark:text-slate-400">Shape: {selectedDistribution.shape?.join(' × ') || '—'}</span>
                  </div>
                  <div className="h-[340px]">
                    {buildHistogramOption(selectedDistribution, textColor, gridColor) && (
                      <ReactECharts option={buildHistogramOption(selectedDistribution, textColor, gridColor)} style={{ height: '100%', width: '100%' }} />
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'saliency' && (
          <div className="card shadow-md border-theme-border flex flex-col gap-5">
            <div>
              <h3 className="h3 text-theme-text-primary">Saliency Maps</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                Probe examples are replayed against the latest captured model state so you can see which pixels or features drive the current prediction.
              </p>
            </div>

            {!saliency?.available ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-6 py-10 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/30 dark:text-slate-400">
                {saliency?.reason || 'Saliency data is not available for this run yet.'}
              </div>
            ) : saliency.modality === 'vision' ? (
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
                {saliency.samples.map((sample) => (
                  <div key={sample.index} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950/60">
                    <div className="flex items-center justify-between gap-3 mb-4">
                      <div>
                        <p className="text-sm font-semibold text-theme-text-primary">Sample {sample.index + 1}</p>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          Target {sample.target ?? '—'} · Prediction {sample.prediction ?? '—'}
                        </p>
                      </div>
                      {sample.confidence != null && (
                        <span className="rounded-full border border-fuchsia-500/20 bg-fuchsia-500/10 px-2.5 py-1 text-xs text-fuchsia-400">
                          {(Number(sample.confidence) * 100).toFixed(1)}% confidence
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Input</p>
                        <HeatmapGrid values={sample.input} tone="input" />
                      </div>
                      <div>
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Saliency</p>
                        <HeatmapGrid values={sample.saliency} tone="saliency" />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.2fr)_minmax(320px,0.8fr)] gap-5">
                <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950/60">
                  <div className="flex items-center justify-between gap-3 mb-3">
                    <div>
                      <p className="text-sm font-semibold text-theme-text-primary">Most Influential Features</p>
                      <p className="text-xs text-slate-500 dark:text-slate-400">Mean absolute saliency across the latest probe examples.</p>
                    </div>
                    <Sparkles className="h-4 w-4 text-fuchsia-400" />
                  </div>
                  <div className="h-[360px]">
                    <ReactECharts
                      option={buildStructuredSaliencyOption(saliency.feature_importance, textColor, gridColor)}
                      style={{ height: '100%', width: '100%' }}
                    />
                  </div>
                </div>
                <div className="space-y-3">
                  {saliency.samples.slice(0, 4).map((sample) => (
                    <div key={sample.index} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-800 dark:bg-slate-950/60">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold text-theme-text-primary">Sample {sample.index + 1}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            Target {sample.target ?? '—'} · Prediction {sample.prediction ?? '—'}
                          </p>
                        </div>
                        {sample.confidence != null && (
                          <span className="rounded-full border border-fuchsia-500/20 bg-fuchsia-500/10 px-2.5 py-1 text-xs text-fuchsia-400">
                            {(Number(sample.confidence) * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                      <div className="mt-3 text-xs font-mono text-slate-300">
                        <p className="text-slate-500 mb-1">Input</p>
                        <p className="truncate">{sample.input.slice(0, 8).map((value) => formatNumber(value, 3)).join(', ')}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'embeddings' && (
          <div className="card shadow-md border-theme-border flex flex-col gap-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="h3 text-theme-text-primary">Embedding Projections</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                  Latest probe activations are projected into 2D with PCA so you can inspect whether classes separate cleanly or collapse together.
                </p>
              </div>
              {selectedEmbedding && (
                <div className="rounded-xl border border-sky-500/20 bg-sky-500/10 px-3 py-2 text-xs text-sky-400">
                  {selectedEmbedding.pooling === 'none' ? 'Direct projection' : `Pooling: ${selectedEmbedding.pooling.replace('_', ' ')}`}
                </div>
              )}
            </div>

            {!embeddings?.available ? (
              <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 px-6 py-10 text-sm text-slate-500 dark:border-slate-800 dark:bg-slate-900/30 dark:text-slate-400">
                {embeddings?.reason || 'Embedding projections are not available for this run yet.'}
              </div>
            ) : (
              <>
                <div className="flex flex-wrap gap-3">
                  <select
                    value={selectedEmbedding?.layer || ''}
                    onChange={(event) => setEmbeddingLayer(event.target.value)}
                    className="min-w-[260px] rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-200"
                  >
                    {embeddings.layers.map((layer) => (
                      <option key={layer.layer} value={layer.layer}>
                        {layer.layer}
                      </option>
                    ))}
                  </select>
                  <div className="flex gap-2">
                    {['target', 'prediction'].map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setEmbeddingColorMode(mode)}
                        className={`rounded-lg px-3 py-2 text-sm border transition ${
                          embeddingColorMode === mode
                            ? 'border-theme-primary bg-theme-primary/10 text-theme-primary'
                            : 'border-slate-200 dark:border-slate-800 text-theme-text-secondary'
                        }`}
                      >
                        Color by {mode}
                      </button>
                    ))}
                  </div>
                </div>

                {selectedEmbedding && (
                  <>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="card">
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Samples</p>
                        <p className="text-lg font-mono text-theme-text-primary">{selectedEmbedding.matrix_shape?.[0] || '—'}</p>
                      </div>
                      <div className="card">
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">Feature Width</p>
                        <p className="text-lg font-mono text-theme-text-primary">{selectedEmbedding.matrix_shape?.[1] || '—'}</p>
                      </div>
                      <div className="card">
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">PC 1 Variance</p>
                        <p className="text-lg font-mono text-theme-text-primary">
                          {((selectedEmbedding.explained_variance_ratio?.[0] || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="card">
                        <p className="text-xs uppercase tracking-wider text-slate-500 mb-2">PC 2 Variance</p>
                        <p className="text-lg font-mono text-theme-text-primary">
                          {((selectedEmbedding.explained_variance_ratio?.[1] || 0) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <div className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-800 dark:bg-slate-950/60">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <div>
                          <p className="text-sm font-semibold text-theme-text-primary">{selectedEmbedding.layer}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">
                            Original activation shape: {selectedEmbedding.original_shape?.join(' × ') || '—'}
                          </p>
                        </div>
                        <span className="rounded-full border border-sky-500/20 bg-sky-500/10 px-2.5 py-1 text-xs text-sky-400">
                          Probe step {embeddings.step}
                        </span>
                      </div>
                      <div className="h-[420px]">
                        <ReactECharts
                          option={buildEmbeddingOption(selectedEmbedding, embeddings, embeddingColorMode, textColor, gridColor)}
                          style={{ height: '100%', width: '100%' }}
                        />
                      </div>
                    </div>
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
