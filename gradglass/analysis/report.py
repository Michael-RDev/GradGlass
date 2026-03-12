from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from gradglass.analysis.registry import TestResult, TestStatus

@dataclass
class PostRunReport:
    run_id: str
    generated_at: str
    checkpoint_diff_summary: dict = field(default_factory=dict)
    gradient_flow_analysis: dict = field(default_factory=dict)
    training_metrics_summary: dict = field(default_factory=dict)
    artifact_store_summary: dict = field(default_factory=dict)
    tests: dict = field(default_factory=dict)
    summary_text: str = ''

    def to_dict(self):
        return {'run_id': self.run_id, 'generated_at': self.generated_at, 'checkpoint_diff_summary': self.checkpoint_diff_summary, 'gradient_flow_analysis': self.gradient_flow_analysis, 'training_metrics_summary': self.training_metrics_summary, 'artifact_store_summary': self.artifact_store_summary, 'tests': self.tests}

    def save(self, run_dir):
        analysis_dir = run_dir / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / 'report.json', 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        with open(analysis_dir / 'summary.txt', 'w') as f:
            f.write(self.summary_text)
        with open(analysis_dir / 'tests.jsonl', 'w') as f:
            for test_result in self.tests.get('results', []):
                entry = dict(test_result)
                entry['timestamp'] = self.generated_at
                f.write(json.dumps(entry, default=str) + '\n')

    @classmethod
    def from_file(cls, run_dir):
        report_path = run_dir / 'analysis' / 'report.json'
        if not report_path.exists():
            return None
        with open(report_path) as f:
            data = json.load(f)
        summary_path = run_dir / 'analysis' / 'summary.txt'
        summary_text = ''
        if summary_path.exists():
            summary_text = summary_path.read_text()
        return cls(run_id=data.get('run_id', ''), generated_at=data.get('generated_at', ''), checkpoint_diff_summary=data.get('checkpoint_diff_summary', {}), gradient_flow_analysis=data.get('gradient_flow_analysis', {}), training_metrics_summary=data.get('training_metrics_summary', {}), artifact_store_summary=data.get('artifact_store_summary', {}), tests=data.get('tests', {}), summary_text=summary_text)

    @classmethod
    def generate(cls, run_id, store, run_dir, tests='all', save=True, print_summary=True):
        from gradglass.analysis.runner import AnalysisRunner
        runner = AnalysisRunner(run_id=run_id, store=store, run_dir=run_dir)
        ctx = runner.build_context()
        test_results = runner.run_all(tests=tests)
        sections = runner.generate_summary_sections(ctx)
        passed = sum((1 for t in test_results if t.status == TestStatus.PASS))
        warned = sum((1 for t in test_results if t.status == TestStatus.WARN))
        failed = sum((1 for t in test_results if t.status == TestStatus.FAIL))
        skipped = sum((1 for t in test_results if t.status == TestStatus.SKIP))
        tests_dict = {'passed': passed, 'warned': warned, 'failed': failed, 'skipped': skipped, 'total': len(test_results), 'results': [t.to_dict() for t in test_results]}
        summary_text = runner.render_text(sections, test_results)
        report = cls(run_id=run_id, generated_at=time.strftime('%Y-%m-%d %H:%M:%S'), checkpoint_diff_summary=sections.get('checkpoint_diff_summary', {}), gradient_flow_analysis=sections.get('gradient_flow_analysis', {}), training_metrics_summary=sections.get('training_metrics_summary', {}), artifact_store_summary=sections.get('artifact_store_summary', {}), tests=tests_dict, summary_text=summary_text)
        if save:
            report.save(run_dir)
        if print_summary:
            print(summary_text)
        return report
