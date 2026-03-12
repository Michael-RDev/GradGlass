import { create } from 'zustand';
import { fetchRun, fetchMetrics, fetchAlerts, createMetricsStream } from '../api';

const useRunStore = create((set, get) => ({
  activeRunId: null,
  metadata: null,
  metrics: [],
  alerts: [],
  ws: null,

  setActiveRun: async (runId) => {
    const { ws } = get();
    if (ws) {
      ws.close();
    }
    
    set({ activeRunId: runId, metadata: null, metrics: [], alerts: [], ws: null });

    try {
      const [metaData, metricsData, alertsData] = await Promise.all([
        fetchRun(runId),
        fetchMetrics(runId),
        fetchAlerts(runId)
      ]);

      set({
        metadata: metaData,
        metrics: metricsData?.metrics || [],
        alerts: alertsData?.alerts || []
      });

      const newWs = createMetricsStream(runId, (message) => {
        if (message.type === 'metrics_update') {
          set((state) => ({
            metrics: [...state.metrics, ...message.data]
          }));
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
    const keys = new Set();
    metrics.forEach(m => {
      Object.keys(m).forEach(k => {
        if (k !== 'step' && k !== 'timestamp') {
          keys.add(k);
        }
      });
    });
    return Array.from(keys).sort();
  }
}));

export default useRunStore;
