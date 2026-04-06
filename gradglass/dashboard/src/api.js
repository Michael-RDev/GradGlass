const API_BASE = '/api';

function encodeRunId(runId) {
  return encodeURIComponent(runId);
}

async function fetchJSON(url) {
  const res = await fetch(`${API_BASE}${url}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function fetchRuns() {
  return fetchJSON('/runs');
}

export async function fetchRun(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}`);
}

export async function fetchMetrics(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/metrics`);
}

export async function fetchOverview(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/overview`);
}

export async function fetchCheckpoints(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/checkpoints`);
}

export async function fetchAlerts(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/alerts`);
}

export async function fetchCompare(runIds) {
  return fetchJSON(`/compare?run_ids=${runIds.join(',')}`);
}

export async function fetchDiff(runId, stepA, stepB, includeDeltas = false) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/diff?a=${stepA}&b=${stepB}&include_deltas=${includeDeltas}`);
}

export async function fetchGradients(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/gradients`);
}

export async function fetchActivations(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/activations`);
}

export async function fetchDistributions(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/distributions`);
}

export async function fetchSaliency(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/saliency`);
}

export async function fetchShap(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/shap`);
}

export async function fetchLime(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/lime`);
}

export async function fetchEmbeddings(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/embeddings`);
}

export async function fetchPredictions(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/predictions`);
}

export async function fetchArchitecture(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/architecture`);
}

export async function fetchAnalysis(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/analysis`);
}

export async function fetchDataMonitor(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/data-monitor`);
}

export async function fetchLeakageReport(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/leakage`);
}

export async function fetchEvalLab(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/eval`);
}

export async function fetchInfrastructureTelemetry(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/infrastructure`);
}

export function createMetricsStream(runId, onMessage) {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${window.location.host}/api/runs/${encodeRunId(runId)}/stream`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  ws.onerror = (err) => console.error('WebSocket error:', err);
  return ws;
}

export async function fetchFreezeCode(runId) {
  return fetchJSON(`/runs/${encodeRunId(runId)}/freeze_code`);
}
