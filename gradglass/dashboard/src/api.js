const API_BASE = '/api';

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
  return fetchJSON(`/runs/${runId}`);
}

export async function fetchMetrics(runId) {
  return fetchJSON(`/runs/${runId}/metrics`);
}

export async function fetchOverview(runId) {
  return fetchJSON(`/runs/${runId}/overview`);
}

export async function fetchCheckpoints(runId) {
  return fetchJSON(`/runs/${runId}/checkpoints`);
}

export async function fetchAlerts(runId) {
  return fetchJSON(`/runs/${runId}/alerts`);
}

export async function fetchCompare(runIds) {
  return fetchJSON(`/compare?run_ids=${runIds.join(',')}`);
}

export async function fetchDiff(runId, stepA, stepB, includeDeltas = false) {
  return fetchJSON(`/runs/${runId}/diff?a=${stepA}&b=${stepB}&include_deltas=${includeDeltas}`);
}

export async function fetchGradients(runId) {
  return fetchJSON(`/runs/${runId}/gradients`);
}

export async function fetchActivations(runId) {
  return fetchJSON(`/runs/${runId}/activations`);
}

export async function fetchPredictions(runId) {
  return fetchJSON(`/runs/${runId}/predictions`);
}

export async function fetchArchitecture(runId) {
  return fetchJSON(`/runs/${runId}/architecture`);
}

export async function fetchAnalysis(runId) {
  return fetchJSON(`/runs/${runId}/analysis`);
}

export async function fetchLeakageReport(runId) {
  return fetchJSON(`/runs/${runId}/leakage`);
}

export async function fetchEvalLab(runId) {
  return fetchJSON(`/runs/${runId}/eval`);
}

export function createMetricsStream(runId, onMessage) {
  const ws = new WebSocket(`ws://${window.location.host}/api/runs/${runId}/stream`);
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    onMessage(data);
  };
  ws.onerror = (err) => console.error('WebSocket error:', err);
  return ws;
}

export async function fetchFreezeCode(runId) {
  return fetchJSON(`/runs/${runId}/freeze_code`);
}
