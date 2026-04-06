import React from 'react';

export function formatInterpretabilityValue(value) {
  if (value == null) return '—';
  if (Array.isArray(value)) {
    return value.map((item) => formatInterpretabilityValue(item)).join(', ');
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function buildStructuredSaliencyOption(featureImportance, textColor, gridColor) {
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

export function buildAttentionHeatmapOption(tokens, matrix, textColor, theme) {
  const rows = Array.isArray(matrix) ? matrix : [];
  const heatmapData = [];
  rows.forEach((row, rowIndex) => {
    (Array.isArray(row) ? row : []).forEach((value, columnIndex) => {
      heatmapData.push([columnIndex, rowIndex, Number(value) || 0]);
    });
  });

  return {
    animation: false,
    tooltip: { position: 'top' },
    grid: { left: 72, right: 24, top: 24, bottom: 72 },
    xAxis: {
      type: 'category',
      data: tokens,
      axisLabel: { interval: 0, rotate: 45, color: textColor },
    },
    yAxis: {
      type: 'category',
      data: tokens,
      axisLabel: { color: textColor },
    },
    visualMap: {
      min: 0,
      max: 1,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 8,
      textStyle: { color: textColor },
      inRange: {
        color: theme === 'dark' ? ['#1e293b', '#f59e0b', '#ef4444'] : ['#e2e8f0', '#fb923c', '#dc2626'],
      },
    },
    series: [
      {
        type: 'heatmap',
        data: heatmapData,
        emphasis: {
          itemStyle: {
            shadowBlur: 8,
            shadowColor: 'rgba(15, 23, 42, 0.35)',
          },
        },
      },
    ],
  };
}

export function HeatmapGrid({ values, tone = 'input' }) {
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
            style={{ backgroundColor: baseColor(Number(value) || 0) }}
            title={`${rowIndex}, ${colIndex}: ${Number(value).toFixed(3)}`}
          />
        ))
      )}
    </div>
  );
}
