from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Optional, Union
from gradglass.analysis.registry import TestCategory, TestContext, TestRegistry, TestResult, TestSeverity, TestStatus
import gradglass.analysis.builtins

class AnalysisRunner:

    def __init__(self, run_id, store, run_dir=None):
        self.run_id = run_id
        self.store = store
        self.run_dir = run_dir or store.get_run_dir(run_id)

    def build_context(self):
        ctx = TestContext(run_id=self.run_id, run_dir=self.run_dir, store=self.store)
        ctx.metadata = self.store.get_run_metadata(self.run_id)
        ctx.metrics = self.store.get_metrics(self.run_id)
        ctx.checkpoints_meta = self.store.list_checkpoints(self.run_id)
        ctx.architecture = self.store.get_architecture(self.run_id)
        ctx.gradient_summaries = self.store.get_gradient_summaries(self.run_id)
        ctx.activation_stats = self.store.get_activation_stats(self.run_id)
        ctx.predictions = self.store.get_predictions(self.run_id)
        ctx.distributed_info = self.store.get_distributed_info(self.run_id)
        ctx.rank_list = self.store.list_ranks(self.run_id)
        return ctx

    def run_all(self, tests='all'):
        ctx = self.build_context()
        registry = TestRegistry.all_tests()
        if tests == 'all':
            test_ids = list(registry.keys())
        else:
            test_ids = tests
        results = []
        for test_id in test_ids:
            registered = registry.get(test_id)
            if registered is None:
                results.append(TestResult(id=test_id, title=f'Unknown test: {test_id}', status=TestStatus.SKIP, severity=TestSeverity.LOW, category=TestCategory.ARTIFACT, details={'error': 'Test not found in registry'}))
                continue
            start = time.time()
            try:
                result = registered.fn(ctx)
                result.duration_ms = (time.time() - start) * 1000
            except Exception as e:
                result = TestResult(id=test_id, title=registered.title, status=TestStatus.FAIL, severity=registered.severity, category=registered.category, details={'error': str(e), 'type': type(e).__name__}, recommendation='Test crashed. Check artifacts integrity.', duration_ms=(time.time() - start) * 1000)
            results.append(result)
        return results

    def generate_summary_sections(self, ctx=None):
        if ctx is None:
            ctx = self.build_context()
        sections = {}
        ckpt_summary = {'checkpoints_saved': 0, 'checkpoints': [], 'diff': None}
        if ctx.has_checkpoints:
            ckpt_summary['checkpoints_saved'] = len(ctx.checkpoints_meta)
            for ck in ctx.checkpoints_meta:
                ckpt_summary['checkpoints'].append({'step': ck['step'], 'tag': ck.get('tag'), 'num_params': ck.get('num_params', 0), 'size_mb': ck.get('size_mb', 0)})
            if len(ctx.checkpoint_steps()) >= 2:
                try:
                    from gradglass.diff import full_diff
                    steps = ctx.checkpoint_steps()
                    w_a = ctx.load_checkpoint(steps[0])
                    w_b = ctx.load_checkpoint(steps[-1])
                    diff_result = full_diff(w_a, w_b, self.run_id, steps[0], steps[-1])
                    top_layers = []
                    sorted_layers = sorted(diff_result.layers, key=lambda d: d.frob_norm, reverse=True)
                    for lr in sorted_layers[:5]:
                        top_layers.append({'layer': lr.layer_name, 'frob_norm': round(lr.frob_norm, 4), 'cos_sim': round(lr.cos_sim, 4), 'severity': lr.severity.value})
                    ckpt_summary['diff'] = {'step_a': steps[0], 'step_b': steps[-1], 'layers_compared': diff_result.summary['total_layers'], 'severity_counts': {'low': diff_result.summary.get('low', 0), 'medium': diff_result.summary.get('medium', 0), 'high': diff_result.summary.get('high', 0), 'critical': diff_result.summary.get('critical', 0)}, 'top_changed_layers': top_layers}
                except Exception:
                    pass
        sections['checkpoint_diff_summary'] = ckpt_summary
        grad_summary = {'has_data': False, 'layers_tracked': 0, 'healthy': 0, 'flagged': 0, 'flagged_layers': []}
        if ctx.has_grad_summaries:
            from gradglass.diff import gradient_flow_analysis
            analysis = gradient_flow_analysis(ctx.gradient_summaries)
            flagged = [a for a in analysis if a['flags']]
            healthy = [a for a in analysis if not a['flags']]
            grad_summary = {'has_data': True, 'layers_tracked': len(analysis), 'healthy': len(healthy), 'flagged': len(flagged), 'flagged_layers': [{'layer': a['layer'], 'flags': a['flags'], 'grad_norm': round(a['grad_norm'], 6)} for a in flagged]}
        sections['gradient_flow_analysis'] = grad_summary
        metrics_summary = {'has_data': False}
        if ctx.has_metrics:
            losses = [m['loss'] for m in ctx.metrics if 'loss' in m]
            accs = [m['acc'] for m in ctx.metrics if 'acc' in m]
            lrs = [m['lr'] for m in ctx.metrics if 'lr' in m]
            metrics_summary = {'has_data': True, 'total_steps': len(ctx.metrics), 'loss_start': round(losses[0], 4) if losses else None, 'loss_final': round(losses[-1], 4) if losses else None, 'acc_start': round(accs[0] * 100, 1) if accs else None, 'acc_final': round(accs[-1] * 100, 1) if accs else None, 'lr_start': lrs[0] if lrs else None, 'lr_final': lrs[-1] if lrs else None}
        sections['training_metrics_summary'] = metrics_summary
        total_bytes = sum((p.stat().st_size for p in self.run_dir.rglob('*') if p.is_file()))
        sections['artifact_store_summary'] = {'run_id': self.run_id, 'path': str(self.run_dir), 'storage_mb': round(total_bytes / (1024 * 1024), 1), 'storage_bytes': total_bytes}
        return sections

    def render_text(self, sections, test_results):
        lines = []

        def section(title):
            lines.append(f"\n{'=' * 60}")
            lines.append(f'  {title}')
            lines.append(f"{'=' * 60}")
        ckpt = sections.get('checkpoint_diff_summary', {})
        section('Checkpoint Diff Summary')
        lines.append(f"  Checkpoints saved: {ckpt.get('checkpoints_saved', 0)}")
        for c in ckpt.get('checkpoints', []):
            tag = f"  [{c['tag']}]" if c.get('tag') else ''
            lines.append(f"    step={c['step']}{tag}  params={c.get('num_params', 0):,}  size={c.get('size_mb', '?')} MB")
        diff = ckpt.get('diff')
        if diff:
            lines.append(f"\n  Diff: step {diff['step_a']}  →  step {diff['step_b']}")
            lines.append(f"  Layers compared : {diff['layers_compared']}")
            sc = diff.get('severity_counts', {})
            lines.append(f"  LOW severity    : {sc.get('low', 0)}")
            lines.append(f"  MEDIUM severity : {sc.get('medium', 0)}")
            lines.append(f"  HIGH severity   : {sc.get('high', 0)}")
            lines.append(f"  CRITICAL        : {sc.get('critical', 0)}")
            top = diff.get('top_changed_layers', [])
            if top:
                lines.append(f'\n  Top {len(top)} most-changed layers:')
                for lr in top:
                    lines.append(f"    {lr['layer']:<35} frob={lr['frob_norm']:.4f}  cos={lr['cos_sim']:.4f}  [{lr['severity']}]")
        grad = sections.get('gradient_flow_analysis', {})
        section('Gradient Flow Analysis')
        if grad.get('has_data'):
            lines.append(f"  Layers tracked  : {grad['layers_tracked']}")
            lines.append(f"  Healthy         : {grad['healthy']}")
            lines.append(f"  Flagged         : {grad['flagged']}")
            for fl in grad.get('flagged_layers', []):
                lines.append(f"    {fl['layer']:<35} flags={fl['flags']}  norm={fl['grad_norm']:.4e}")
            if not grad.get('flagged_layers'):
                lines.append('  ✓ No vanishing / exploding gradient issues detected.')
        else:
            lines.append('  (no gradient summaries found — try increasing epochs or lowering `every`)')
        met = sections.get('training_metrics_summary', {})
        section('Training Metrics Summary')
        if met.get('has_data'):
            lines.append(f"  Total steps logged : {met['total_steps']}")
            if met.get('loss_final') is not None:
                lines.append(f"  Final loss         : {met['loss_final']:.4f}  (start: {met.get('loss_start', '?')})")
            if met.get('acc_final') is not None:
                lines.append(f"  Final batch acc    : {met['acc_final']:.1f}%  (start: {met.get('acc_start', '?')}%)")
            if met.get('lr_final') is not None:
                lines.append(f"  Final LR           : {met['lr_final']}")
        else:
            lines.append('  (no metrics found)')
        store = sections.get('artifact_store_summary', {})
        section('Artifact Store')
        lines.append(f"  Run ID   : {store.get('run_id', '?')}")
        lines.append(f"  Path     : {store.get('path', '?')}")
        lines.append(f"  Storage  : {store.get('storage_mb', 0)} MB")
        section('Test Suite Results')
        passed = sum((1 for t in test_results if t.status == TestStatus.PASS))
        warned = sum((1 for t in test_results if t.status == TestStatus.WARN))
        failed = sum((1 for t in test_results if t.status == TestStatus.FAIL))
        skipped = sum((1 for t in test_results if t.status == TestStatus.SKIP))
        lines.append(f'  Total    : {len(test_results)}')
        lines.append(f'  ✅ Pass  : {passed}')
        lines.append(f'  ⚠️  Warn  : {warned}')
        lines.append(f'  ❌ Fail  : {failed}')
        lines.append(f'  ⏭  Skip  : {skipped}')
        if failed:
            lines.append(f'\n  Failed tests:')
            for t in test_results:
                if t.status == TestStatus.FAIL:
                    lines.append(f'    ❌ {t.id}: {t.title}')
                    if t.recommendation:
                        lines.append(f'       → {t.recommendation}')
        if warned:
            lines.append(f'\n  Warnings:')
            for t in test_results:
                if t.status == TestStatus.WARN:
                    lines.append(f'    ⚠️  {t.id}: {t.title}')
                    if t.recommendation:
                        lines.append(f'       → {t.recommendation}')
        return '\n'.join(lines)
