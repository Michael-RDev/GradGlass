from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np


class ArtifactStore:
    DEFAULT_ROOT = Path.cwd() / "gg_artifacts"

    def __init__(self, root=None):
        self.root = Path(root) if root else self.DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)

    def ensure_run_dir(self, run_id):
        run_dir = self.root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["checkpoints", "gradients", "activations", "predictions", "probes", "slices"]:
            (run_dir / subdir).mkdir(exist_ok=True)
        return run_dir

    def get_run_dir(self, run_id):
        return self.root / "runs" / run_id

    def get_run_metadata(self, run_id):
        meta_path = self.root / "runs" / run_id / "metadata.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def get_runtime_state(self, run_id):
        state_path = self.root / "runs" / run_id / "runtime_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    def list_runs(self):
        runs = []
        runs_dir = self.root / "runs"
        if not runs_dir.exists():
            return runs
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["run_id"] = run_dir.name
                total_bytes = sum((p.stat().st_size for p in run_dir.rglob("*") if p.is_file()))
                meta["storage_bytes"] = total_bytes
                meta["storage_mb"] = round(total_bytes / (1024 * 1024), 1)
                ckpt_dir = run_dir / "checkpoints"
                meta["num_checkpoints"] = len(list(ckpt_dir.glob("step_*.npz"))) if ckpt_dir.exists() else 0
                metrics = self.get_latest_metrics(run_dir.name)
                if metrics:
                    meta["latest_loss"] = metrics.get("loss")
                    meta["latest_acc"] = metrics.get("acc")
                    meta["total_steps"] = metrics.get("step", 0)
                runs.append(meta)
        return runs

    def get_metrics(self, run_id):
        metrics_path = self.root / "runs" / run_id / "metrics.jsonl"
        if not metrics_path.exists():
            return []
        metrics = []
        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
        return metrics

    def get_latest_metrics(self, run_id):
        metrics_path = self.root / "runs" / run_id / "metrics.jsonl"
        if not metrics_path.exists():
            return None
        last_line = None
        with open(metrics_path) as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        if last_line:
            return json.loads(last_line)
        return None

    def list_checkpoints(self, run_id):
        ckpt_dir = self.root / "runs" / run_id / "checkpoints"
        if not ckpt_dir.exists():
            return []
        checkpoints = []
        for meta_file in sorted(ckpt_dir.glob("step_*_meta.json"), key=lambda p: int(p.stem.split("_")[1])):
            with open(meta_file) as f:
                meta = json.load(f)
            step = meta["step"]
            npz_path = ckpt_dir / f"step_{step}.npz"
            meta["has_weights"] = npz_path.exists()
            if npz_path.exists():
                meta["size_bytes"] = npz_path.stat().st_size
                meta["size_mb"] = round(npz_path.stat().st_size / (1024 * 1024), 2)
            checkpoints.append(meta)
        return checkpoints

    def load_checkpoint(self, run_id, step):
        ckpt_dir = self.root / "runs" / run_id / "checkpoints"
        # Try NumPy checkpoint first.
        npz_path = ckpt_dir / f"step_{step}.npz"
        if npz_path.exists():
            data = np.load(str(npz_path))
            return dict(data)
        # Fall back to legacy pickle-style checkpoints.
        pkl_path = ckpt_dir / f"step_{step}.pkl"
        if pkl_path.exists():
            try:
                import joblib
                return {'model': joblib.load(str(pkl_path))}
            except ImportError:
                import pickle
                with open(pkl_path, 'rb') as fh:
                    return {'model': pickle.load(fh)}
        raise FileNotFoundError(f"Checkpoint not found at step {step} in {ckpt_dir}")

    def get_gradient_summaries(self, run_id):
        grad_dir = self.root / "runs" / run_id / "gradients"
        if not grad_dir.exists():
            return []
        summaries = []
        for f_path in sorted(grad_dir.glob("summaries_step_*.json")):
            try:
                step = int(f_path.stem.split("_")[-1])
                with open(f_path) as f:
                    data = json.load(f)
                summaries.append({"step": step, "layers": data})
            except (ValueError, json.JSONDecodeError):
                continue
        return summaries

    def get_activation_stats(self, run_id):
        act_dir = self.root / "runs" / run_id / "activations"
        if not act_dir.exists():
            return []
        stats = []
        for f_path in sorted(act_dir.glob("*_stats.json")):
            try:
                with open(f_path) as f:
                    data = json.load(f)
                parts = f_path.stem.replace("_stats", "").rsplit("_step_", 1)
                if len(parts) == 2:
                    data["layer"] = parts[0].replace("_", ".")
                    data["step"] = int(parts[1])
                stats.append(data)
            except Exception:
                continue
        for f_path in sorted(act_dir.glob("*.npy")):
            try:
                parts = f_path.stem.rsplit("_step_", 1)
                if len(parts) == 2:
                    arr = np.load(str(f_path))
                    data = {
                        "layer": parts[0].replace("_", "."),
                        "step": int(parts[1]),
                        "mean": float(np.mean(arr)),
                        "var": float(np.var(arr)),
                        "sparsity": float(np.sum(np.abs(arr) < 1e-06) / arr.size),
                        "shape": list(arr.shape),
                    }
                    stats.append(data)
            except Exception:
                continue
        return stats

    def get_predictions(self, run_id):
        pred_dir = self.root / "runs" / run_id / "predictions"
        if not pred_dir.exists():
            return []
        predictions = []
        for f_path in sorted(pred_dir.glob("probe_step_*.json")):
            try:
                with open(f_path) as f:
                    data = json.load(f)
                predictions.append(data)
            except Exception:
                continue
        return predictions

    def list_probe_steps(self, run_id):
        probe_dir = self.root / "runs" / run_id / "probes"
        if not probe_dir.exists():
            return []
        steps = []
        for meta_path in sorted(probe_dir.glob("probe_step_*.json")):
            try:
                steps.append(int(meta_path.stem.split("_")[-1]))
            except ValueError:
                continue
        return sorted(steps)

    def load_probe_bundle(self, run_id, step=None):
        probe_dir = self.root / "runs" / run_id / "probes"
        if not probe_dir.exists():
            raise FileNotFoundError(f"No probe directory found for run '{run_id}'")

        if step is None:
            steps = self.list_probe_steps(run_id)
            if not steps:
                raise FileNotFoundError(f"No probe bundles found for run '{run_id}'")
            step = steps[-1]

        meta_path = probe_dir / f"probe_step_{step}.json"
        data_path = probe_dir / f"probe_step_{step}.npz"
        if not meta_path.exists() or not data_path.exists():
            raise FileNotFoundError(f"Probe bundle missing for run '{run_id}' at step {step}")

        with open(meta_path) as f:
            meta = json.load(f)

        arrays = {}
        with np.load(str(data_path), allow_pickle=False) as data:
            for key in data.files:
                arrays[key] = data[key]

        return {"meta": meta, "arrays": arrays}

    def get_architecture(self, run_id):
        arch_path = self.root / "runs" / run_id / "model_structure.json"
        if not arch_path.exists():
            return None
        with open(arch_path) as f:
            return json.load(f)

    def get_distributed_info(self, run_id):
        dist_path = self.root / "runs" / run_id / "distributed_index.json"
        if not dist_path.exists():
            return None
        with open(dist_path) as f:
            return json.load(f)

    def list_ranks(self, run_id):
        run_dir = self.root / "runs" / run_id
        ranks = []
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and d.name.startswith("rank_"):
                ranks.append(d.name)
        return ranks
