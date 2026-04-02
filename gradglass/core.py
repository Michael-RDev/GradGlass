from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional
from gradglass.browser import open_url_detached
from gradglass.run import Run
from gradglass.artifacts import ArtifactStore


class GradGlass:
    def __init__(self, root=None):
        self.store = ArtifactStore(root=root)
        self.auto_open = os.environ.get("GRADGLASS_OPEN", "").lower() in ("1", "true", "yes")

    def configure(self, auto_open=False, root=None):
        self.auto_open = auto_open
        if root is not None:
            self.store = ArtifactStore(root=root)
        return self

    def run(self, name, auto_open=None, **options):
        should_open = auto_open if auto_open is not None else self.auto_open
        return Run(name=name, store=self.store, auto_open=should_open, **options)

    def list_runs(self):
        runs = []
        runs_dir = self.store.root / "runs"
        if not runs_dir.exists():
            return runs
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["run_id"] = run_dir.name
                    total_bytes = sum((p.stat().st_size for p in run_dir.rglob("*") if p.is_file()))
                    meta["storage_bytes"] = total_bytes
                    meta["storage_mb"] = round(total_bytes / (1024 * 1024), 1)
                    runs.append(meta)
                except (json.JSONDecodeError, OSError):
                    continue
        return runs

    def open_last(self):
        runs = self.list_runs()
        if not runs:
            print("No runs found in artifact store.")
            return
        runs.sort(key=lambda r: r.get("start_time", ""), reverse=True)
        last_run_id = runs[0]["run_id"]
        run = Run.from_existing(last_run_id, store=self.store)
        run.open()

    def get_run(self, run_id):
        return Run.from_existing(run_id, store=self.store)

    def analyze_run(self, run_id, **kwargs):
        run = self.get_run(run_id)
        return run.analyze(**kwargs)

    def monitor_dataset(self, task, dataset_name=None, task_hint=None, config=None, run_dir=None, run_id=None):
        from gradglass.analysis.data_monitor import DatasetMonitorBuilder

        return DatasetMonitorBuilder(
            task=task,
            dataset_name=dataset_name,
            task_hint=task_hint,
            config=config,
            run_dir=run_dir,
            run_id=run_id,
        )

    def test(self):
        from gradglass.analysis.registry import test as test_decorator

        return test_decorator

    def monitor(self, port=8432, open_browser=True):
        from gradglass.server import create_app, start_server

        app = create_app(self.store)
        actual_port = start_server(app, port=port)
        url = f"http://localhost:{actual_port}"
        print(f"\U0001f52c GradGlass dashboard: {url}")
        if open_browser:
            open_url_detached(url)
        return actual_port


gg = GradGlass()
