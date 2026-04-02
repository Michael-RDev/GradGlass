export { clsx } from 'clsx';
export { twMerge } from 'tailwind-merge';

export const DEFAULT_METRIC_EXCLUDE_KEYS = new Set([
  'step',
  'timestamp',
  'fit_duration_s',
  'epoch',
  'epoch_idx',
  'epoch_end',
  'lr',
  'learning_rate',
]);

function toFiniteNumber(value) {
  if (value == null) return null;
  const numeric = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function extractNumericSeries(metrics, key) {
  if (!Array.isArray(metrics) || !key) return [];
  const byStep = new Map();
  for (const row of metrics) {
    if (!row || !(key in row)) continue;
    const step = toFiniteNumber(row.step);
    const value = toFiniteNumber(row[key]);
    if (step == null || value == null) continue;
    byStep.set(step, value);
  }
  return Array.from(byStep.entries()).sort((a, b) => a[0] - b[0]);
}

export function latestSeriesValue(series) {
  if (!Array.isArray(series) || series.length === 0) return null;
  const point = series[series.length - 1];
  return Array.isArray(point) ? point[1] : null;
}

export function discoverMetricKeysInOrder(metrics, excludeKeys = DEFAULT_METRIC_EXCLUDE_KEYS) {
  if (!Array.isArray(metrics)) return [];
  const excluded = new Set(Array.from(excludeKeys || []).map((k) => String(k).toLowerCase()));
  const seen = new Set();
  const keys = [];

  for (const row of metrics) {
    if (!row) continue;
    for (const key of Object.keys(row)) {
      const lowered = key.toLowerCase();
      if (excluded.has(lowered) || seen.has(lowered)) continue;
      seen.add(lowered);
      keys.push(key);
    }
  }

  return keys;
}

function includesAnyToken(value, tokens) {
  return tokens.some((token) => value.includes(token));
}

export function metricSemanticRelevanceScore(key) {
  const lowered = String(key || '').toLowerCase();
  if (!lowered) return 0;

  let score = 0;
  if (includesAnyToken(lowered, ['loss', 'logloss', 'nll', 'rmse', 'mse', 'mae', 'error'])) score += 500;
  if (includesAnyToken(lowered, ['accuracy', '_acc', 'acc_', 'precision', 'recall', 'f1', 'auc', 'r2', 'score'])) score += 450;
  if (includesAnyToken(lowered, ['val', 'validation', 'test'])) score += 30;
  if (includesAnyToken(lowered, ['train'])) score += 15;
  if (includesAnyToken(lowered, ['duration', 'latency', 'seconds', '_s', 'time'])) score -= 200;

  return score;
}

export function rankMetricKeys(
  metrics,
  {
    excludeKeys = DEFAULT_METRIC_EXCLUDE_KEYS,
  } = {}
) {
  const orderedKeys = discoverMetricKeysInOrder(metrics, excludeKeys);
  const ranked = orderedKeys
    .map((key, orderIndex) => ({
      key,
      orderIndex,
      semantic: metricSemanticRelevanceScore(key),
      points: extractNumericSeries(metrics, key).length,
    }))
    .filter((item) => item.points > 0);

  ranked.sort((a, b) => {
    if (b.semantic !== a.semantic) return b.semantic - a.semantic;
    if (b.points !== a.points) return b.points - a.points;
    return a.orderIndex - b.orderIndex;
  });

  return ranked.map((item) => item.key);
}

export function resolvePrimaryMetricKey(
  metrics,
  {
    priorityKeys = [],
    includeTokens = [],
    excludeTokens = [],
    excludeKeys = DEFAULT_METRIC_EXCLUDE_KEYS,
  } = {}
) {
  const allKeys = discoverMetricKeysInOrder(metrics, excludeKeys);
  const seriesByLowerKey = new Map();
  const keyByLowerKey = new Map();

  for (const key of allKeys) {
    const series = extractNumericSeries(metrics, key);
    if (series.length === 0) continue;
    const lowered = key.toLowerCase();
    seriesByLowerKey.set(lowered, series);
    keyByLowerKey.set(lowered, key);
  }

  if (seriesByLowerKey.size === 0) return null;

  for (const preferred of priorityKeys) {
    const lowered = String(preferred).toLowerCase();
    if (seriesByLowerKey.has(lowered)) return keyByLowerKey.get(lowered) || null;
  }

  const normalizedIncludeTokens = includeTokens.map((token) => String(token).toLowerCase()).filter(Boolean);
  const normalizedExcludeTokens = excludeTokens.map((token) => String(token).toLowerCase()).filter(Boolean);

  const matchesTokenPolicy = (loweredKey) => {
    if (normalizedExcludeTokens.some((token) => loweredKey.includes(token))) return false;
    if (normalizedIncludeTokens.length === 0) return true;
    return normalizedIncludeTokens.some((token) => loweredKey.includes(token));
  };

  for (const key of allKeys) {
    const lowered = key.toLowerCase();
    if (!seriesByLowerKey.has(lowered)) continue;
    if (!matchesTokenPolicy(lowered)) continue;
    return keyByLowerKey.get(lowered) || key;
  }

  const [fallbackLowerKey] = seriesByLowerKey.keys();
  return keyByLowerKey.get(fallbackLowerKey) || null;
}

export function formatMetricKeyLabel(key) {
  if (!key) return 'Metric';

  const normalized = String(key)
    .replace(/[_\-.]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!normalized) return 'Metric';

  return normalized
    .split(' ')
    .map((token) => {
      const lower = token.toLowerCase();
      if (lower === 'lr') return 'LR';
      if (lower === 'auc') return 'AUC';
      if (lower === 'f1') return 'F1';
      if (lower === 'r2') return 'R2';
      if (lower === 'rmse') return 'RMSE';
      if (lower === 'mse') return 'MSE';
      if (lower === 'mae') return 'MAE';
      if (lower === 'kl') return 'KL';
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(' ');
}
