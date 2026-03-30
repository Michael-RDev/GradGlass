import { create } from 'zustand';
import { fetchRun, fetchMetrics, fetchAlerts, fetchOverview, createMetricsStream } from '../api';
import { DEFAULT_METRIC_EXCLUDE_KEYS, rankMetricKeys } from '../utils';

const useRunStore = create((set, get) => ({
  activeRunId: null,
  metadata: null,
  metrics: [],
  alerts: [],
  alertsSummary: null,
  overview: null,
  ws: null,

  setActiveRun: async (runId) => {
    const { ws } = get();
    if (ws) {
      ws.close();
    }
    
    set({ activeRunId: runId, metadata: null, metrics: [], alerts: [], alertsSummary: null, overview: null, ws: null });

    try {
      const [metaData, metricsData, alertsData, overviewData] = await Promise.all([
        fetchRun(runId),
        fetchMetrics(runId),
        fetchAlerts(runId),
        fetchOverview(runId)
      ]);

      set({
        metadata: metaData,
        metrics: metricsData?.metrics || [],
        alerts: alertsData?.alerts || [],
        alertsSummary: alertsData?.summary || null,
        overview: overviewData || null
      });

      const newWs = createMetricsStream(runId, (message) => {
        if (message.type === 'metrics_update') {
          set((state) => ({
            metrics: [...state.metrics, ...message.data]
          }));
        } else if (message.type === 'overview_update') {
          set({ overview: message.data || null });
        } else if (message.type === 'alerts_update') {
          set({
            alerts: message.data?.alerts || [],
            alertsSummary: message.data?.summary || null,
          });
        }
      });

      set({ ws: newWs });
    } catch (err) {
      console.error('Failed to load run data:', err);
    }
  },

  discoverMetricKeys: () => {
    const { metrics } = get();
    if (!metrics || metrics.length === 0) return [];
    return rankMetricKeys(metrics, { excludeKeys: DEFAULT_METRIC_EXCLUDE_KEYS });
  }
}));

export default useRunStore;
