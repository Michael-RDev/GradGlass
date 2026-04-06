import test from 'node:test';
import assert from 'node:assert/strict';

import {
  buildHardExamples,
  buildInterpretabilityCards,
  buildInterpretabilityTabs,
  createInterpretabilityViewModel,
} from './interpretabilityUtils.js';

test('buildInterpretabilityTabs hides attention unless explicit artifacts exist', () => {
  assert.deepEqual(buildInterpretabilityTabs({}).map((tab) => tab.id), ['attribution', 'shap', 'other-tools', 'hard-examples']);
  assert.deepEqual(
    buildInterpretabilityTabs({ analysis: { interpretability: { attention: { available: true, tokens: ['a'], matrix: [[1]] } } } }).map((tab) => tab.id),
    ['attribution', 'shap', 'other-tools', 'hard-examples', 'attention']
  );
});

test('buildInterpretabilityCards extracts attribution signals from analysis results', () => {
  const cards = buildInterpretabilityCards({
    tests: {
      results: [
        {
          id: 'GRAD_INPUT_SALIENCY',
          status: 'pass',
          severity: 'MEDIUM',
          details: {
            top_10_by_gradient_norm: [{ layer: 'encoder.block.3', mean_grad_norm: 0.4321, steps_captured: 8 }],
          },
        },
        {
          id: 'SHAP_GRAD_ATTRIBUTION_RANK',
          status: 'warn',
          severity: 'MEDIUM',
          details: { rank_overlap: 1 },
        },
      ],
    },
  });

  assert.equal(cards[0].title, 'Gradient Saliency');
  assert.match(cards[0].summary, /encoder\.block\.3/);
  assert.equal(cards[1].status, 'warn');
  assert.match(cards[1].summary, /1\/3/);
  assert.equal(cards[2].status, 'skip');
});

test('buildHardExamples falls back to label flips when confidence is missing', () => {
  const hardExamples = buildHardExamples({
    predictions: [
      {
        step: 10,
        y_true: [0, 1, 2, 3],
        y_pred: [0, 1, 2, 2],
      },
      {
        step: 20,
        y_true: [0, 1, 2, 3],
        y_pred: [1, 1, 0, 2],
      },
    ],
  });

  assert.equal(hardExamples.length, 3);
  assert.equal(hardExamples[0].index, 0);
  assert.equal(hardExamples[0].regressed, true);
  assert.equal(hardExamples[1].index, 2);
  assert.equal(hardExamples[1].changed, true);
  assert.equal(hardExamples[2].index, 3);
});

test('createInterpretabilityViewModel combines cards, tabs, and hard examples', () => {
  const viewModel = createInterpretabilityViewModel({
    analysis: {
      tests: {
        results: [
          {
            id: 'LIME_PROXY_CONFIDENCE',
            status: 'pass',
            severity: 'MEDIUM',
            details: { mean: 0.76, std: 0.12, samples: 16 },
          },
        ],
      },
    },
    predictions: {
      predictions: [
        { y_true: [0, 1], y_pred: [1, 1], confidence: [0.91, 0.55] },
      ],
    },
  });

  assert.deepEqual(viewModel.tabs.map((tab) => tab.id), ['attribution', 'shap', 'other-tools', 'hard-examples']);
  assert.equal(viewModel.cards[2].status, 'pass');
  assert.equal(viewModel.hardExamples[0].index, 0);
});
