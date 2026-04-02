import { create } from 'zustand';
import { fetchRun, fetchMetrics, fetchAlerts, fetchOverview, createMetricsStream } from '../api';
import { DEFAULT_METRIC_EXCLUDE_KEYS, rankMetricKeys } from '../utils';

function closeSocket(ws) {
  if (!ws) return;
  try {
    ws.close();
  } catch {
    // Ignore websocket shutdown failures during route transitions.
  }
}

function normalizeStatus(status) {
  const normalized = (status || '').toString().trim().toLowerCase();
  if (normalized === 'complete' || normalized === 'finished') return 'completed';
  return normalized;
}

function isTerminalStatus(status) {
  return ['completed', 'failed', 'cancelled', 'interrupted'].includes(normalizeStatus(status));
}

function metricIdentity(metric) {
  if (!metric || typeof metric !== 'object') return null;
  const step = metric.step ?? 'none';
  const timestamp = metric.timestamp ?? 'none';
  return `${step}::${timestamp}`;
}

function mergeMetricRows(existing, incoming) {
  const merged = [];
  const seen = new Set();

  for (const row of [...(existing || []), ...(incoming || [])]) {
    const id = metricIdentity(row);
    if (id && seen.has(id)) {
      continue;
    }
    if (id) {
      seen.add(id);
    }
    merged.push(row);
  }

  merged.sort((a, b) => {
    const stepDelta = (a?.step ?? 0) - (b?.step ?? 0);
    if (stepDelta !== 0) return stepDelta;
    return (a?.timestamp ?? 0) - (b?.timestamp ?? 0);
  });

  return merged;
}

const useRunStore = create((set, get) => ({
  activeRunId: null,
  metadata: null,
  metrics: [],
  alerts: [],
  alertsSummary: null,
  overview: null,
  ws: null,
  loadVersion: 0,

  setActiveRun: async (runId) => {
    const { ws, loadVersion } = get();
    closeSocket(ws);

    const nextLoadVersion = loadVersion + 1;
    set({
      activeRunId: runId,
      metadata: null,
      metrics: [],
      alerts: [],
      alertsSummary: null,
      overview: null,
      ws: null,
      loadVersion: nextLoadVersion,
    });

    try {
      const [metaData, metricsData, alertsData, overviewData] = await Promise.all([
        fetchRun(runId),
        fetchMetrics(runId),
        fetchAlerts(runId),
        fetchOverview(runId)
      ]);

      if (get().loadVersion !== nextLoadVersion || get().activeRunId !== runId) {
        return;
      }

      set({
        metadata: metaData,
        metrics: mergeMetricRows([], metricsData?.metrics || []),
        alerts: alertsData?.alerts || [],
        alertsSummary: alertsData?.summary || null,
        overview: overviewData || null
      });

      if (isTerminalStatus(overviewData?.status || metaData?.status)) {
        return;
      }

      const newWs = createMetricsStream(runId, (message) => {
        if (get().activeRunId !== runId) {
          return;
        }
        if (message.type === 'metrics_update') {
          set((state) => ({
            metrics: mergeMetricRows(state.metrics, message.data)
          }));
        } else if (message.type === 'overview_update') {
          const nextOverview = message.data || null;
          if (isTerminalStatus(nextOverview?.status)) {
            const currentWs = get().ws;
            closeSocket(currentWs);
            set({ overview: nextOverview, ws: null });
            return;
          }
          set({ overview: nextOverview });
        } else if (message.type === 'alerts_update') {
          set({
            alerts: message.data?.alerts || [],
            alertsSummary: message.data?.summary || null,
          });
        }
      });

      if (get().loadVersion !== nextLoadVersion || get().activeRunId !== runId) {
        closeSocket(newWs);
        return;
      }

      set({ ws: newWs });
    } catch (err) {
      if (get().loadVersion === nextLoadVersion && get().activeRunId === runId) {
        console.error('Failed to load run data:', err);
      }
    }
  },

  clearActiveRun: () => {
    const { ws, loadVersion } = get();
    closeSocket(ws);
    set({
      activeRunId: null,
      metadata: null,
      metrics: [],
      alerts: [],
      alertsSummary: null,
      overview: null,
      ws: null,
      loadVersion: loadVersion + 1,
    });
  },

  discoverMetricKeys: () => {
    const { metrics } = get();
    if (!metrics || metrics.length === 0) return [];
    return rankMetricKeys(metrics, { excludeKeys: DEFAULT_METRIC_EXCLUDE_KEYS });
  }
}));

export default useRunStore;
