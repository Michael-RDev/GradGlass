export const DATA_MONITOR_PREFERENCE_KEY = 'gradglass:data-monitor:prefs:v1';

export const DEFAULT_PANEL_ORDER = [
  'pipeline',
  'composition',
  'split-comparisons',
  'leakage',
  'recommendations',
];

export const DEFAULT_PANEL_VISIBILITY = {
  pipeline: true,
  composition: true,
  'split-comparisons': true,
  leakage: true,
  recommendations: true,
};

export const DEFAULT_COLLAPSED_SECTIONS = {
  pipeline: false,
  composition: false,
  'split-comparisons': false,
  leakage: false,
  recommendations: false,
};

export const DEFAULT_FILTERS = {
  split: 'all',
  stage: 'all',
  category: 'all',
  severity: 'all',
  status: 'all',
  modality: 'all',
};

export function normalizeDataMonitorPreferences(raw = {}) {
  const panelOrder = Array.isArray(raw.panelOrder) ? raw.panelOrder.filter(Boolean) : [];
  const uniquePanelOrder = [...new Set([...panelOrder, ...DEFAULT_PANEL_ORDER])].filter((panel) =>
    DEFAULT_PANEL_ORDER.includes(panel)
  );

  return {
    panelOrder: uniquePanelOrder,
    panelVisibility: {
      ...DEFAULT_PANEL_VISIBILITY,
      ...(raw.panelVisibility && typeof raw.panelVisibility === 'object' ? raw.panelVisibility : {}),
    },
    collapsedSections: {
      ...DEFAULT_COLLAPSED_SECTIONS,
      ...(raw.collapsedSections && typeof raw.collapsedSections === 'object' ? raw.collapsedSections : {}),
    },
    defaultFilters: {
      ...DEFAULT_FILTERS,
      ...(raw.defaultFilters && typeof raw.defaultFilters === 'object' ? raw.defaultFilters : {}),
    },
  };
}

export function loadDataMonitorPreferences(storage = globalThis?.localStorage) {
  if (!storage || typeof storage.getItem !== 'function') {
    return normalizeDataMonitorPreferences();
  }
  try {
    const raw = storage.getItem(DATA_MONITOR_PREFERENCE_KEY);
    if (!raw) {
      return normalizeDataMonitorPreferences();
    }
    return normalizeDataMonitorPreferences(JSON.parse(raw));
  } catch {
    return normalizeDataMonitorPreferences();
  }
}

export function saveDataMonitorPreferences(preferences, storage = globalThis?.localStorage) {
  if (!storage || typeof storage.setItem !== 'function') {
    return;
  }
  storage.setItem(DATA_MONITOR_PREFERENCE_KEY, JSON.stringify(normalizeDataMonitorPreferences(preferences)));
}

export function buildInitialFilters(routeMode, defaultFilters = DEFAULT_FILTERS) {
  const filters = { ...DEFAULT_FILTERS, ...defaultFilters };
  if (routeMode === 'leakage' && filters.category === 'all') {
    filters.category = 'leakage';
  }
  return filters;
}

export function movePanel(panelOrder, panelId, direction) {
  const next = [...panelOrder];
  const currentIndex = next.indexOf(panelId);
  if (currentIndex === -1) return next;
  const swapIndex = direction === 'up' ? currentIndex - 1 : currentIndex + 1;
  if (swapIndex < 0 || swapIndex >= next.length) {
    return next;
  }
  [next[currentIndex], next[swapIndex]] = [next[swapIndex], next[currentIndex]];
  return next;
}

export function buildVisiblePanelOrder(preferences, routeMode = 'data') {
  const normalized = normalizeDataMonitorPreferences(preferences);
  const ordered = normalized.panelOrder.filter((panel) => normalized.panelVisibility[panel] !== false);
  if (routeMode !== 'leakage') {
    return ordered;
  }
  const leakageFirst = ordered.filter((panel) => panel === 'leakage');
  const recommendations = ordered.filter((panel) => panel === 'recommendations');
  const rest = ordered.filter((panel) => panel !== 'leakage' && panel !== 'recommendations');
  return [...leakageFirst, ...recommendations, ...rest];
}

export function filterChecks(checks = [], filters = DEFAULT_FILTERS) {
  return checks.filter((check) => {
    if (filters.category !== 'all' && check.category !== filters.category) return false;
    if (filters.severity !== 'all' && check.severity !== filters.severity) return false;
    if (filters.status !== 'all' && check.status !== filters.status) return false;
    return true;
  });
}

export function filterStageSnapshots(snapshots = [], filters = DEFAULT_FILTERS) {
  return snapshots.filter((snapshot) => {
    if (filters.split !== 'all' && snapshot.split !== filters.split) return false;
    if (filters.stage !== 'all' && snapshot.stage !== filters.stage) return false;
    if (filters.status !== 'all' && snapshot.status !== filters.status) return false;
    if (filters.modality !== 'all') {
      const modalities = snapshot?.modality_metadata?.modalities || [];
      if (!modalities.includes(filters.modality)) return false;
    }
    return true;
  });
}

export function filterCompositionPanels(panels = [], filters = DEFAULT_FILTERS) {
  return panels.filter((panel) => {
    if (filters.split !== 'all' && panel.split && panel.split !== filters.split) return false;
    if (filters.modality !== 'all' && panel.type === 'modality-breakdown') {
      const series = panel?.data?.series || {};
      if (!(filters.modality in series)) return false;
    }
    return true;
  });
}

export function filterSplitComparisonPanels(panels = [], filters = DEFAULT_FILTERS) {
  return panels.filter((panel) => {
    if (filters.split === 'all') return true;
    const splitA = panel?.data?.split_a;
    const splitB = panel?.data?.split_b;
    return splitA === filters.split || splitB === filters.split;
  });
}
