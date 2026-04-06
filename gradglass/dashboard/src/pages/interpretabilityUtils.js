export const INTERPRETABILITY_TESTS = [
  {
    id: 'GRAD_INPUT_SALIENCY',
    title: 'Gradient Saliency',
    emptySummary: 'Gradient summaries are needed to estimate which layers receive the strongest learning signal.',
    fallbackSeverity: 'MEDIUM',
  },
  {
    id: 'SHAP_GRAD_ATTRIBUTION_RANK',
    title: 'Attribution Stability',
    emptySummary: 'More gradient checkpoints are needed before attribution rankings can be compared over time.',
    fallbackSeverity: 'MEDIUM',
  },
  {
    id: 'LIME_PROXY_CONFIDENCE',
    title: 'Confidence Spread',
    emptySummary: 'Prediction probes with confidence values are needed to judge whether confidence varies meaningfully.',
    fallbackSeverity: 'MEDIUM',
  },
];

function valuesEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function toFixedMaybe(value, digits = 4) {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(value).toFixed(digits);
}

function summarizeTestResult(test, config) {
  if (!test) {
    return config.emptySummary;
  }

  const details = test.details || {};
  if (test.id === 'GRAD_INPUT_SALIENCY') {
    const topLayer = details.top_10_by_gradient_norm?.[0];
    if (topLayer) {
      return `Top layer ${topLayer.layer} with mean grad norm ${toFixedMaybe(topLayer.mean_grad_norm, 4)} across ${topLayer.steps_captured} steps.`;
    }
  }

  if (test.id === 'SHAP_GRAD_ATTRIBUTION_RANK') {
    if (details.rank_overlap != null) {
      return `Top-layer overlap ${details.rank_overlap}/3 between early and late training snapshots.`;
    }
  }

  if (test.id === 'LIME_PROXY_CONFIDENCE') {
    if (details.mean != null && details.std != null) {
      const sampleCount = details.samples != null ? ` across ${details.samples} samples` : '';
      return `Mean confidence ${toFixedMaybe(details.mean, 3)} with std ${toFixedMaybe(details.std, 3)}${sampleCount}.`;
    }
  }

  if (details.reason) return details.reason;
  if (test.recommendation) return test.recommendation;
  return config.emptySummary;
}

export function buildInterpretabilityCards(analysis) {
  const results = analysis?.tests?.results || [];
  return INTERPRETABILITY_TESTS.map((config) => {
    const test = results.find((item) => item.id === config.id);
    return {
      id: config.id,
      title: config.title,
      status: test?.status || 'skip',
      severity: test?.severity || config.fallbackSeverity,
      summary: summarizeTestResult(test, config),
      recommendation: test?.recommendation || null,
      details: test?.details || {},
    };
  });
}

export function extractAttentionPayload(payloads = {}) {
  const candidates = [
    payloads.attention,
    payloads.saliency?.attention,
    payloads.analysis?.attention,
    payloads.analysis?.interpretability?.attention,
    payloads.analysis?.tests?.attention,
  ].filter(Boolean);

  return (
    candidates.find((payload) => {
      if (payload?.available === true) return true;
      if (Array.isArray(payload?.matrix) && payload.matrix.length) return true;
      if (Array.isArray(payload?.weights) && payload.weights.length) return true;
      if (Array.isArray(payload?.heads) && payload.heads.length) return true;
      if (Array.isArray(payload?.layers) && payload.layers.length) return true;
      return false;
    }) || null
  );
}

export function buildInterpretabilityTabs(payloads = {}) {
  const tabs = [
    { id: 'attribution', label: 'Attribution' },
    { id: 'shap', label: 'SHAP Analysis' },
    { id: 'other-tools', label: 'Explainability Tools' },
    { id: 'hard-examples', label: 'Hard Examples' },
  ];
  if (extractAttentionPayload(payloads)) {
    tabs.push({ id: 'attention', label: 'Attention' });
  }
  return tabs;
}

export function buildHardExamples(predictionPayload = {}) {
  const records = predictionPayload?.predictions || [];
  const latest = records.at(-1);
  const previous = records.length > 1 ? records.at(-2) : null;
  if (!latest || !Array.isArray(latest.y_true) || !Array.isArray(latest.y_pred)) {
    return [];
  }

  const latestConfidence = Array.isArray(latest.confidence) ? latest.confidence : [];
  const previousConfidence = Array.isArray(previous?.confidence) ? previous.confidence : [];
  const previousPredictions = Array.isArray(previous?.y_pred) ? previous.y_pred : [];
  const previousTargets = Array.isArray(previous?.y_true) ? previous.y_true : [];
  const total = Math.min(latest.y_true.length, latest.y_pred.length);
  const examples = [];

  for (let index = 0; index < total; index += 1) {
    const trueLabel = latest.y_true[index];
    const prediction = latest.y_pred[index];
    const confidence = latestConfidence[index] ?? null;
    const prevPrediction = previousPredictions[index];
    const prevTarget = previousTargets[index];
    const prevConfidence = previousConfidence[index] ?? null;
    const isCorrect = valuesEqual(trueLabel, prediction);
    const changed = prevPrediction !== undefined && !valuesEqual(prevPrediction, prediction);
    const prevCorrect = prevPrediction !== undefined && prevTarget !== undefined ? valuesEqual(prevTarget, prevPrediction) : null;
    const regressed = prevCorrect === true && !isCorrect;
    const confidenceDelta =
      prevConfidence == null || confidence == null ? null : Math.abs(Number(confidence) - Number(prevConfidence));
    const unstable = changed || regressed || (confidenceDelta != null && confidenceDelta >= 0.15);

    if (!isCorrect || unstable) {
      let reason = 'Prediction changed since the previous probe.';
      if (!isCorrect && confidence != null) {
        reason = `${(Number(confidence) * 100).toFixed(1)}% confident miss on the latest logged probe.`;
      } else if (!isCorrect) {
        reason = regressed ? 'This sample regressed from correct to incorrect.' : 'Incorrect prediction on the latest logged probe.';
      } else if (regressed) {
        reason = 'This sample regressed relative to the previous logged probe.';
      } else if (confidenceDelta != null) {
        reason = `Confidence shifted by ${(confidenceDelta * 100).toFixed(1)} percentage points.`;
      }

      examples.push({
        index,
        trueLabel,
        prediction,
        confidence,
        prevPrediction: prevPrediction ?? null,
        prevConfidence,
        isCorrect,
        changed,
        regressed,
        confidenceDelta,
        reason,
        step: latest.step ?? null,
      });
    }
  }

  return examples
    .sort((left, right) => {
      if (left.isCorrect !== right.isCorrect) return left.isCorrect ? 1 : -1;
      if (left.confidence != null || right.confidence != null) {
        if (left.confidence == null) return 1;
        if (right.confidence == null) return -1;
        if (Number(left.confidence) !== Number(right.confidence)) return Number(right.confidence) - Number(left.confidence);
      }
      if (left.regressed !== right.regressed) return left.regressed ? -1 : 1;
      if (left.changed !== right.changed) return left.changed ? -1 : 1;
      if ((right.confidenceDelta ?? -1) !== (left.confidenceDelta ?? -1)) {
        return (right.confidenceDelta ?? -1) - (left.confidenceDelta ?? -1);
      }
      return left.index - right.index;
    })
    .slice(0, 24);
}

export function createInterpretabilityViewModel(payloads = {}) {
  return {
    tabs: buildInterpretabilityTabs(payloads),
    cards: buildInterpretabilityCards(payloads.analysis),
    hardExamples: buildHardExamples(payloads.predictions),
    attention: extractAttentionPayload(payloads),
    shap: payloads.shap || null,
    lime: payloads.lime || null,
  };
}
