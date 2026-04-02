import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildInitialFilters,
  buildVisiblePanelOrder,
  filterChecks,
  filterCompositionPanels,
  filterStageSnapshots,
  loadDataMonitorPreferences,
  movePanel,
  normalizeDataMonitorPreferences,
  saveDataMonitorPreferences,
} from './dataMonitorUtils.js';


function createStorage() {
  const values = new Map();
  return {
    getItem(key) {
      return values.has(key) ? values.get(key) : null;
    },
    setItem(key, value) {
      values.set(key, String(value));
    },
  };
}


test('normalizeDataMonitorPreferences fills missing defaults', () => {
  const preferences = normalizeDataMonitorPreferences({
    panelOrder: ['leakage', 'pipeline'],
    panelVisibility: { pipeline: false },
  });

  assert.equal(preferences.panelOrder[0], 'leakage');
  assert.equal(preferences.panelVisibility.pipeline, false);
  assert.equal(preferences.panelVisibility.composition, true);
  assert.equal(preferences.defaultFilters.category, 'all');
});


test('preferences round-trip through storage', () => {
  const storage = createStorage();
  saveDataMonitorPreferences(
    {
      panelOrder: ['recommendations', 'pipeline'],
      panelVisibility: { recommendations: true, pipeline: false },
      collapsedSections: { pipeline: true },
      defaultFilters: { severity: 'HIGH' },
    },
    storage
  );

  const loaded = loadDataMonitorPreferences(storage);
  assert.equal(loaded.panelOrder[0], 'recommendations');
  assert.equal(loaded.panelVisibility.pipeline, false);
  assert.equal(loaded.collapsedSections.pipeline, true);
  assert.equal(loaded.defaultFilters.severity, 'HIGH');
});


test('buildVisiblePanelOrder prioritizes leakage in leakage mode', () => {
  const preferences = normalizeDataMonitorPreferences({
    panelOrder: ['pipeline', 'composition', 'recommendations', 'leakage'],
  });

  assert.deepEqual(buildVisiblePanelOrder(preferences, 'data'), ['pipeline', 'composition', 'recommendations', 'leakage', 'split-comparisons']);
  assert.deepEqual(buildVisiblePanelOrder(preferences, 'leakage'), ['leakage', 'recommendations', 'pipeline', 'composition', 'split-comparisons']);
});


test('movePanel reorders panels safely', () => {
  assert.deepEqual(movePanel(['pipeline', 'leakage', 'recommendations'], 'leakage', 'up'), ['leakage', 'pipeline', 'recommendations']);
  assert.deepEqual(movePanel(['pipeline', 'leakage', 'recommendations'], 'pipeline', 'up'), ['pipeline', 'leakage', 'recommendations']);
});


test('filters apply to checks, stage snapshots, and composition panels', () => {
  const filters = buildInitialFilters('leakage', {
    split: 'train',
    severity: 'HIGH',
    status: 'failed',
    category: 'leakage',
  });

  const checks = filterChecks(
    [
      { category: 'leakage', severity: 'HIGH', status: 'failed', name: 'A' },
      { category: 'composition', severity: 'HIGH', status: 'failed', name: 'B' },
      { category: 'leakage', severity: 'LOW', status: 'warning', name: 'C' },
    ],
    filters
  );
  assert.equal(checks.length, 1);
  assert.equal(checks[0].name, 'A');

  const snapshots = filterStageSnapshots(
    [
      { split: 'train', stage: 'splitting', status: 'failed', modality_metadata: { modalities: ['tabular'] } },
      { split: 'test', stage: 'splitting', status: 'passed', modality_metadata: { modalities: ['text'] } },
    ],
    { ...filters, stage: 'splitting' }
  );
  assert.equal(snapshots.length, 1);

  const panels = filterCompositionPanels(
    [
      { split: 'train', data: { series: { text: 1 } } },
      { split: 'test', data: { series: { audio: 1 } } },
    ],
    { ...filters, split: 'train' }
  );
  assert.equal(panels.length, 1);
});
