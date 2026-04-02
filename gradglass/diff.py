from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np

class Severity(str, Enum):
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'
    CRITICAL = 'CRITICAL'

@dataclass
class DiffResult:
    layer_name: str
    frob_norm: float
    cos_sim: float
    percent_changed: float
    max_delta: float
    severity: Severity
    delta: Optional[np.ndarray] = field(default=None, repr=False)
    shape: tuple = ()

    def to_dict(self, include_delta=False):
        d = {'name': self.layer_name, 'frob_norm': round(self.frob_norm, 6), 'cos_sim': round(self.cos_sim, 6), 'percent_changed': round(self.percent_changed, 4), 'max_delta': round(self.max_delta, 6), 'severity': self.severity.value, 'shape': list(self.shape)}
        if include_delta and self.delta is not None:
            d['delta_histogram'] = compute_histogram(self.delta)
            d['top_k_deltas'] = top_k_deltas(self.delta, k=10)
        return d

@dataclass
class FullDiffResult:
    run_id: str
    step_a: int
    step_b: int
    layers: list[DiffResult]
    summary: dict

    def to_dict(self, include_deltas=False):
        return {'run_id': self.run_id, 'step_a': self.step_a, 'step_b': self.step_b, 'layers': [lr.to_dict(include_delta=include_deltas) for lr in self.layers], 'summary': self.summary}

def weight_diff(w_a, w_b, layer_name='', threshold=0.0001):
    delta = w_a.astype(np.float64) - w_b.astype(np.float64)
    flat_a = w_a.flatten().astype(np.float64)
    flat_b = w_b.flatten().astype(np.float64)
    fro_norm = float(np.linalg.norm(delta))
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    if norm_a > 0 and norm_b > 0:
        cos_sim = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))
    else:
        cos_sim = 1.0
    percent_changed = float(np.sum(np.abs(delta) > threshold) / max(delta.size, 1))
    max_delta = float(np.max(np.abs(delta)))
    severity = classify_severity(fro_norm, cos_sim, percent_changed)
    return DiffResult(layer_name=layer_name, frob_norm=fro_norm, cos_sim=cos_sim, percent_changed=percent_changed, max_delta=max_delta, severity=severity, delta=delta, shape=tuple(w_a.shape))

def classify_severity(fro_norm, cos_sim, percent_changed):
    severities = []
    if fro_norm > 0.3:
        severities.append(Severity.CRITICAL)
    elif fro_norm > 0.15:
        severities.append(Severity.HIGH)
    elif fro_norm > 0.05:
        severities.append(Severity.MEDIUM)
    else:
        severities.append(Severity.LOW)
    if cos_sim < 0.9:
        severities.append(Severity.CRITICAL)
    elif cos_sim < 0.97:
        severities.append(Severity.HIGH)
    elif cos_sim < 0.995:
        severities.append(Severity.MEDIUM)
    else:
        severities.append(Severity.LOW)
    if percent_changed > 0.5:
        severities.append(Severity.CRITICAL)
    elif percent_changed > 0.2:
        severities.append(Severity.HIGH)
    elif percent_changed > 0.05:
        severities.append(Severity.MEDIUM)
    else:
        severities.append(Severity.LOW)
    priority = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
    return max(severities, key=lambda s: priority[s])

def full_diff(weights_a, weights_b, run_id='', step_a=0, step_b=0, threshold=0.0001):
    layer_diffs = []
    common_layers = set(weights_a.keys()) & set(weights_b.keys())
    added_layers = set(weights_b.keys()) - set(weights_a.keys())
    removed_layers = set(weights_a.keys()) - set(weights_b.keys())
    for layer_name in sorted(common_layers):
        w_a = weights_a[layer_name]
        w_b = weights_b[layer_name]
        if w_a.shape != w_b.shape:
            layer_diffs.append(DiffResult(layer_name=layer_name, frob_norm=float('inf'), cos_sim=0.0, percent_changed=1.0, max_delta=float('inf'), severity=Severity.CRITICAL, shape=tuple(w_a.shape)))
        else:
            diff = weight_diff(w_a, w_b, layer_name=layer_name, threshold=threshold)
            layer_diffs.append(diff)
    priority = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}
    layer_diffs.sort(key=lambda d: priority[d.severity], reverse=True)
    severity_counts = {s.value: 0 for s in Severity}
    for d in layer_diffs:
        severity_counts[d.severity.value] += 1
    summary = {'total_layers': len(layer_diffs), 'added_layers': list(added_layers), 'removed_layers': list(removed_layers), **{k.lower(): v for (k, v) in severity_counts.items()}}
    return FullDiffResult(run_id=run_id, step_a=step_a, step_b=step_b, layers=layer_diffs, summary=summary)

def gradient_flow_analysis(gradient_summaries):
    if not gradient_summaries:
        return []
    all_layers = set()
    for entry in gradient_summaries:
        all_layers.update(entry.get('layers', {}).keys())
    results = []
    for layer_name in sorted(all_layers):
        history = []
        for entry in gradient_summaries:
            if layer_name in entry.get('layers', {}):
                data = entry['layers'][layer_name]
                data['step'] = entry['step']
                history.append(data)
        if not history:
            continue
        latest = history[-1]
        earliest = history[0]
        grad_mean = latest.get('mean', 0)
        grad_var = latest.get('var', 0)
        grad_max = latest.get('max', 0)
        grad_norm = latest.get('norm', 0)
        kl_div = latest.get('kl_div', 0)
        flags = []
        # VANISHING: mean is near-zero AND norm is also near-zero.
        # Requiring both prevents false positives on converged models where
        # signed gradients cancel (mean ≈ 0) but the norm stays healthy.
        if abs(grad_mean) < 1e-07 and grad_norm < 1e-05:
            flags.append('VANISHING')
        # EXPLODING: catch both large-mean and large-norm cases.
        if abs(grad_mean) > 100 or grad_norm > 1000:
            flags.append('EXPLODING')
        if grad_var > 0 and abs(grad_mean) / (grad_var ** 0.5 + 1e-12) < 0.01:
            flags.append('NOISY')
        if kl_div > 0.5:
            flags.append('DISTRIBUTION_SHIFT')
        if grad_max < 1e-08:
            flags.append('DEAD')
        stability_status = 'healthy'
        status_reason = 'Gradient norm is in a healthy range.'
        if 'EXPLODING' in flags or grad_norm > 10:
            stability_status = 'too_large'
            status_reason = 'Gradient norm is unusually large and may indicate unstable updates.'
        elif 'VANISHING' in flags or 'DEAD' in flags or grad_norm < 1e-05:
            stability_status = 'too_small'
            status_reason = 'Gradient norm is very small and may indicate slow or stalled learning.'
        elif 'NOISY' in flags:
            stability_status = 'noisy'
            status_reason = 'Gradient direction appears noisy relative to its variance.'
        result = {
            'layer': layer_name,
            'grad_mean': grad_mean,
            'grad_var': grad_var,
            'grad_max': grad_max,
            'grad_norm': grad_norm,
            'kl_div': kl_div,
            'flags': flags,
            'num_steps': len(history),
            'stability_status': stability_status,
            'stability_reason': status_reason,
            'history': [{'step': h['step'], 'mean': h.get('mean', 0), 'norm': h.get('norm', 0)} for h in history],
        }
        results.append(result)
    return results

def _ks_2samp_numpy(a: np.ndarray, b: np.ndarray):
    """NumPy-only two-sample KS statistic (no p-value). Used as scipy fallback."""
    a_s = np.sort(a)
    b_s = np.sort(b)
    combined = np.sort(np.concatenate([a_s, b_s]))
    cdf_a = np.searchsorted(a_s, combined, side='right') / len(a_s)
    cdf_b = np.searchsorted(b_s, combined, side='right') / len(b_s)
    return float(np.max(np.abs(cdf_a - cdf_b))), float('nan')


def activation_diff(acts_a, acts_b, layer_name=''):
    flat_a = acts_a.flatten()
    flat_b = acts_b.flatten()
    try:
        from scipy import stats as scipy_stats
        (ks_stat, ks_pval) = scipy_stats.ks_2samp(flat_a, flat_b)
    except ImportError:
        (ks_stat, ks_pval) = _ks_2samp_numpy(flat_a, flat_b)
    eps = 1e-06
    sparsity_a = float(np.sum(np.abs(flat_a) < eps) / max(flat_a.size, 1))
    sparsity_b = float(np.sum(np.abs(flat_b) < eps) / max(flat_b.size, 1))
    channel_mse = None
    if acts_a.ndim >= 3 and acts_a.shape == acts_b.shape:
        axes = tuple((i for i in range(acts_a.ndim) if i != 1))
        channel_mse = np.mean((acts_a - acts_b) ** 2, axis=axes).tolist()
    return {'layer': layer_name, 'ks_statistic': float(ks_stat), 'ks_pvalue': float(ks_pval), 'sparsity_a': sparsity_a, 'sparsity_b': sparsity_b, 'sparsity_delta': sparsity_b - sparsity_a, 'channel_mse': channel_mse}

def prediction_diff(pred_a, pred_b):
    result = {'step_a': pred_a.get('step'), 'step_b': pred_b.get('step')}
    y_pred_a = pred_a.get('y_pred', [])
    y_pred_b = pred_b.get('y_pred', [])
    if y_pred_a and y_pred_b:
        min_len = min(len(y_pred_a), len(y_pred_b))
        a = np.array(y_pred_a[:min_len])
        b = np.array(y_pred_b[:min_len])
        if a.ndim == 1 and b.ndim == 1:
            label_flips = int(np.sum(a != b))
            result['label_flips'] = label_flips
            result['flip_rate'] = label_flips / max(min_len, 1)
    conf_a = pred_a.get('confidence', [])
    conf_b = pred_b.get('confidence', [])
    if conf_a and conf_b:
        min_len = min(len(conf_a), len(conf_b))
        result['mean_confidence_a'] = float(np.mean(conf_a[:min_len]))
        result['mean_confidence_b'] = float(np.mean(conf_b[:min_len]))
        result['confidence_delta'] = result['mean_confidence_b'] - result['mean_confidence_a']
    logits_a = pred_a.get('logits_sample', [])
    logits_b = pred_b.get('logits_sample', [])
    if logits_a and logits_b:
        la = np.array(logits_a)
        lb = np.array(logits_b)
        if la.shape == lb.shape:
            result['logit_l2'] = float(np.linalg.norm(la - lb))
    return result

def architecture_diff(arch_a, arch_b):
    layers_a = {l['id']: l for l in arch_a.get('layers', [])}
    layers_b = {l['id']: l for l in arch_b.get('layers', [])}
    ids_a = set(layers_a.keys())
    ids_b = set(layers_b.keys())
    added = ids_b - ids_a
    removed = ids_a - ids_b
    common = ids_a & ids_b
    type_changed = []
    shape_changed = []
    param_changed = []
    for lid in common:
        la = layers_a[lid]
        lb = layers_b[lid]
        if la.get('type') != lb.get('type'):
            type_changed.append({'id': lid, 'old': la['type'], 'new': lb['type']})
        if la.get('output_shape') != lb.get('output_shape'):
            shape_changed.append({'id': lid, 'old_shape': la.get('output_shape'), 'new_shape': lb.get('output_shape')})
        if la.get('param_count') != lb.get('param_count'):
            param_changed.append({'id': lid, 'old_count': la.get('param_count'), 'new_count': lb.get('param_count')})
    edges_a = set((tuple(e) for e in arch_a.get('edges', [])))
    edges_b = set((tuple(e) for e in arch_b.get('edges', [])))
    return {'added_layers': [layers_b[lid] for lid in added], 'removed_layers': [layers_a[lid] for lid in removed], 'type_changed': type_changed, 'shape_changed': shape_changed, 'param_changed': param_changed, 'added_edges': [list(e) for e in edges_b - edges_a], 'removed_edges': [list(e) for e in edges_a - edges_b], 'is_identical': len(added) == 0 and len(removed) == 0 and (len(type_changed) == 0)}

def compute_histogram(tensor, bins=50):
    flat = tensor.flatten()
    (counts, bin_edges) = np.histogram(flat, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}

def top_k_deltas(delta, k=10):
    flat = np.abs(delta.flatten())
    if flat.size <= k:
        top_indices = np.argsort(flat)[::-1]
    else:
        top_indices = np.argpartition(flat, -k)[-k:]
        top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    results = []
    for idx in top_indices:
        multi_idx = np.unravel_index(idx, delta.shape)
        results.append({'index': list((int(i) for i in multi_idx)), 'value': float(delta.flatten()[idx]), 'abs_value': float(flat[idx])})
    return results
