from __future__ import annotations
import json
import time
import uuid
import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import numpy as np
from gradglass.experiment_tracking import infer_total_steps_from_config

if TYPE_CHECKING:
    from gradglass.artifacts import ArtifactStore

_UNSET = object()


class Run:
    def __init__(self, name, store, auto_open=False, **options):
        self.name = name
        self.store = store
        self.options = options
        self.auto_open = auto_open
        self.run_id = f"{name}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.run_dir = self.store.ensure_run_dir(self.run_id)
        self.step = 0
        self.model = None
        self.optimizer = None
        self.framework = None
        self.hooks = []
        self.capture_config = {}
        self.auto_checkpoint_interval = None
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.runtime_state_file = self.run_dir / "runtime_state.json"
        self.start_time = time.time()
        self.server_process = None
        self.server_port = None
        self.grad_buffer = {}
        self.lock = threading.Lock()
        self._git_commit = self._capture_git_commit()
        self.write_metadata(status="running")
        self._write_runtime_state(
            status="running",
            event="init",
            monitor_enabled=bool(self.options.get("monitor", False)),
            monitor_port=self.options.get("port"),
            current_step=self.step,
            total_steps=infer_total_steps_from_config(self.options),
            fatal_exception=None,
        )

    @staticmethod
    def _capture_git_commit() -> Optional[str]:
        try:
            import subprocess

            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    @classmethod
    def from_existing(cls, run_id, store):
        run = object.__new__(cls)
        run.run_id = run_id
        run.store = store
        run.run_dir = store.root / "runs" / run_id
        run.step = 0
        run.model = None
        run.optimizer = None
        run.framework = None
        run.hooks = []
        run.capture_config = {}
        run.auto_checkpoint_interval = None
        run.auto_open = False
        run.options = {}
        run.metrics_file = run.run_dir / "metrics.jsonl"
        run.runtime_state_file = run.run_dir / "runtime_state.json"
        run.start_time = None
        run.server_process = None
        run.server_port = None
        run.grad_buffer = {}
        run.lock = threading.Lock()
        run._git_commit = None  # not needed for read-only access to existing runs
        meta_path = run.run_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            run.name = meta.get("name", run_id)
            run.framework = meta.get("framework")
            run.start_time = meta.get("start_time_epoch")
            run._git_commit = meta.get("git_commit")
        else:
            run.name = run_id
        return run

    def write_metadata(self, status="running"):
        import sys

        env = {"python_version": sys.version}
        try:
            import torch

            env["torch_version"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        try:
            import tensorflow as tf

            env["tensorflow_version"] = tf.__version__
        except (ImportError, AttributeError):
            pass
        meta = {
            "name": self.name,
            "run_id": self.run_id,
            "framework": self.framework,
            "status": status,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time_epoch": self.start_time,
            "config": self.options,
            "capture_config": self.capture_config,
            "environment": env,
            "git_commit": self._git_commit,
        }
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    def _read_runtime_state(self) -> dict[str, Any]:
        if not self.runtime_state_file.exists():
            return {}
        try:
            with open(self.runtime_state_file) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    @staticmethod
    def _collect_resource_snapshot() -> Optional[dict[str, Any]]:
        try:
            import psutil
        except ImportError:
            return None
        try:
            proc = psutil.Process()
            return {
                "rss_bytes": int(proc.memory_info().rss),
                "cpu_percent": float(proc.cpu_percent(interval=None)),
            }
        except Exception:
            return None

    def _write_runtime_state(
        self,
        *,
        status: Optional[str] = None,
        event: Optional[str] = None,
        monitor_enabled: Optional[bool] = None,
        monitor_port: Optional[int] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None,
        fatal_exception: Any = _UNSET,
        heartbeat_ts: Optional[float] = None,
    ) -> None:
        ts = float(heartbeat_ts if heartbeat_ts is not None else time.time())
        with self.lock:
            state = self._read_runtime_state()
            state["heartbeat_ts"] = ts
            state["current_step"] = int(current_step if current_step is not None else self.step)
            if self.start_time is not None:
                state.setdefault("start_time_epoch", float(self.start_time))
            if event is not None:
                state["last_event"] = event
            if status is not None:
                state["status"] = status
            if monitor_enabled is not None:
                state["monitor_enabled"] = bool(monitor_enabled)
            if monitor_port is not None:
                try:
                    state["monitor_port"] = int(monitor_port)
                except (TypeError, ValueError):
                    pass
            if total_steps is not None:
                if total_steps > 0:
                    state["total_steps"] = int(total_steps)
                elif "total_steps" in state:
                    state.pop("total_steps", None)
            if fatal_exception is not _UNSET:
                state["fatal_exception"] = None if fatal_exception in (None, "") else str(fatal_exception)

            resource = self._collect_resource_snapshot()
            if resource is not None:
                state["resource_tracking_available"] = True
                state["resource"] = resource
                state["resource_updated_ts"] = ts
            else:
                state.setdefault("resource_tracking_available", False)

            with open(self.runtime_state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

    def watch(
        self,
        model,
        optimizer=None,
        activations="auto",
        gradients="summary",
        layers="trainable",
        sample_batches=2,
        every=50,
        monitor=None,
        monitor_port=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.capture_config = {
            "activations": activations,
            "gradients": gradients,
            "layers": layers,
            "sample_batches": sample_batches,
            "every": every,
        }
        self.framework = self.detectframework(model)
        self.write_metadata()
        self._write_runtime_state(
            status="running",
            event="watch",
            current_step=self.step,
            total_steps=self._infer_total_steps_from_options_or_model(),
            fatal_exception=None,
        )
        if self.framework in ("sklearn", "xgboost", "lightgbm"):
            from gradglass.capture import SklearnCaptureAdapter

            self.engine = SklearnCaptureAdapter(model=model, run_dir=self.run_dir, config=self.capture_config)
        else:
            from gradglass.capture import CaptureEngine

            self.engine = CaptureEngine(
                model=model,
                optimizer=optimizer,
                framework=self.framework,
                run_dir=self.run_dir,
                config=self.capture_config,
            )
            self.engine.attach_hooks()
        self.engine.extract_architecture()
        monitor_enabled = bool(self.options.get("monitor", False)) if monitor is None else bool(monitor)
        resolved_monitor_port = self.options.get("port", 0) if monitor_port is None else monitor_port
        self._write_runtime_state(
            event="watch_ready",
            monitor_enabled=monitor_enabled,
            monitor_port=resolved_monitor_port if resolved_monitor_port is not None else 0,
            current_step=self.step,
        )
        if monitor_enabled:
            self.monitor(port=resolved_monitor_port or 0, open_browser=True)
        return self

    def detectframework(self, model):
        type_names = [t.__module__ + "." + t.__qualname__ for t in type(model).__mro__]
        for tn in type_names:
            if "torch" in tn:
                return "pytorch"
            if "keras" in tn or "tensorflow" in tn:
                return "tensorflow"
            if "xgboost" in tn:
                return "xgboost"
            if "lightgbm" in tn:
                return "lightgbm"
            if "sklearn" in tn:
                return "sklearn"
        try:
            from sklearn.base import BaseEstimator

            if isinstance(model, BaseEstimator):
                return "sklearn"
        except ImportError:
            pass
        return "unknown"

    def _infer_total_steps_from_options_or_model(self) -> Optional[int]:
        config_total = infer_total_steps_from_config(self.options)
        if config_total and config_total > 0:
            return config_total

        if self.model is None:
            return None

        for key in ("n_estimators", "max_iter", "num_iterations"):
            value = _safe_positive_int(getattr(self.model, key, None))
            if value is not None:
                return value

        get_params = getattr(self.model, "get_params", None)
        if callable(get_params):
            try:
                params = get_params()
            except Exception:
                params = {}
            if isinstance(params, dict):
                for key in ("n_estimators", "max_iter", "num_iterations"):
                    value = _safe_positive_int(params.get(key))
                    if value is not None:
                        return value

        return None

    def log(self, **metrics):
        self.step += 1
        entry = {"step": self.step, "timestamp": time.time(), **metrics}
        if self.optimizer is not None:
            lr = self.get_lr()
            if lr is not None:
                entry["lr"] = lr
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, default=json_default) + "\n")
        if hasattr(self, "engine") and self.step % self.capture_config.get("every", 50) == 0:
            self.engine.capture_gradients(self.step)
        if self.auto_checkpoint_interval and self.step % self.auto_checkpoint_interval == 0:
            self.checkpoint(step=self.step)
        self._write_runtime_state(event="log", current_step=self.step)

    def log_batch(self, x, y=None, y_pred=None, loss=None):
        self.step += 1
        if hasattr(self, "engine"):
            self.engine.log_batch_predictions(step=self.step, x=x, y=y, y_pred=y_pred, loss=loss)
        self._write_runtime_state(event="log_batch", current_step=self.step)

    def checkpoint(self, step=None, tag=None):
        step = step or self.step
        if not hasattr(self, "engine"):
            raise RuntimeError("Call run.watch(model) before checkpointing.")
        self.engine.save_checkpoint(step=step, tag=tag)

    def checkpoint_every(self, interval):
        self.auto_checkpoint_interval = interval

    def flush(self):
        if hasattr(self, "engine"):
            self.engine.flush_writes()

    def analyze(self, tests="all", print_summary=True):
        from gradglass.analysis.report import PostRunReport

        return PostRunReport.generate(
            run_id=self.run_id,
            store=self.store,
            run_dir=self.run_dir,
            tests=tests,
            save=True,
            print_summary=print_summary,
        )

    def finish(self, open=False, analyze=True, print_summary=True):
        self.flush()
        self._write_runtime_state(status="complete", event="finish", current_step=self.step, fatal_exception=None)
        self.write_metadata(status="complete")
        if hasattr(self, "engine"):
            self.engine.cleanup()
        report = None
        if analyze:
            report = self.analyze(print_summary=print_summary)
        if open:
            if self.server_port is not None:
                # Server already running from monitor(), just open the browser
                url = f"http://localhost:{self.server_port}/?run={self.run_id}"
                webbrowser.open(url)
            else:
                self.open()
        return report

    def fail(self, error: Any, open=False, analyze=False, print_summary=True):
        self.flush()
        self._write_runtime_state(
            status="failed",
            event="fail",
            current_step=self.step,
            fatal_exception=error,
        )
        self.write_metadata(status="failed")
        if hasattr(self, "engine"):
            self.engine.cleanup()
        report = None
        if analyze:
            report = self.analyze(print_summary=print_summary)
        if open:
            if self.server_port is not None:
                webbrowser.open(f"http://localhost:{self.server_port}/?run={self.run_id}")
            else:
                self.open()
        return report

    def open(self):
        import uvicorn
        from gradglass.server import create_app, find_free_port

        app = create_app(self.store)
        port = find_free_port()
        url = f"http://localhost:{port}/?run={self.run_id}"
        print(f"GradGlass dashboard: {url}")
        print("Press Ctrl+C to stop the server.")

        def open_browser_fn():
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=open_browser_fn, daemon=True).start()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

    def serve(self, port=0, open_browser=True):
        from gradglass.server import create_app, start_server

        if self.server_process is None:
            app = create_app(self.store)
            actual_port = start_server(app, port=port)
            self.server_port = actual_port
        if open_browser:
            url = f"http://localhost:{self.server_port}/?run={self.run_id}"
            webbrowser.open(url)
        return self.server_port

    def monitor(self, port=0, open_browser=True):
        """Launch the dashboard server before or during training for live monitoring.

        This starts the server in a background thread and optionally opens the
        browser.  The server stays alive until the process exits or ``finish()``
        is called.

        Returns the port the server is listening on.
        """
        from gradglass.server import create_app, start_server

        if self.server_port is not None:
            # Server already running
            if open_browser:
                url = f"http://localhost:{self.server_port}/?run={self.run_id}"
                webbrowser.open(url)
            self._write_runtime_state(
                event="monitor_reuse",
                monitor_enabled=True,
                monitor_port=self.server_port,
                current_step=self.step,
            )
            return self.server_port

        app = create_app(self.store)
        actual_port = start_server(app, port=port)
        self.server_port = actual_port
        self._write_runtime_state(
            event="monitor_start",
            monitor_enabled=True,
            monitor_port=actual_port,
            current_step=self.step,
        )
        url = f"http://localhost:{actual_port}/?run={self.run_id}"
        print(f"\U0001f52c GradGlass live monitor: {url}")
        if open_browser:

            def open_fn():
                time.sleep(1.0)
                webbrowser.open(url)

            threading.Thread(target=open_fn, daemon=True).start()
        return actual_port

    def check_leakage(self, train_x, train_y, test_x, test_y, max_samples=2000, print_summary=True):
        """Run data leakage detection between train and test sets.

        Accepts numpy arrays or torch Tensors.  Results are saved to the
        run's analysis directory and returned as a ``LeakageReport``.
        """
        import numpy as np

        # Convert torch tensors if needed
        for name, arr in [("train_x", train_x), ("train_y", train_y), ("test_x", test_x), ("test_y", test_y)]:
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            elif hasattr(arr, "numpy"):
                arr = arr.numpy()
            if name == "train_x":
                train_x = np.asarray(arr)
            elif name == "train_y":
                train_y = np.asarray(arr)
            elif name == "test_x":
                test_x = np.asarray(arr)
            elif name == "test_y":
                test_y = np.asarray(arr)

        from gradglass.analysis.leakage import run_leakage_detection

        save_path = self.run_dir / "analysis" / "leakage_report.json"
        report = run_leakage_detection(
            train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y, max_samples=max_samples, save_path=save_path
        )
        if print_summary:
            status = "\u2705 ALL PASSED" if report.passed else f"\u274c {report.num_failed} FAILED"
            print(f"\n\U0001f50d Data Leakage Report: {status}")
            print(
                f"   Checks: {report.num_passed} passed, {report.num_failed} failed  ({report.total_duration_ms:.0f}ms)"
            )
            for r in report.results:
                icon = "\u2705" if r.passed else "\u274c"
                print(f"   {icon} [{r.severity}] {r.title}")
                if not r.passed and r.recommendation:
                    print(f"      \u2192 {r.recommendation}")
            print()
        return report

    def check_leakage_from_loaders(self, train_loader, test_loader, max_samples=2000, print_summary=True):
        import numpy as np

        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for DataLoader leakage checking")

        def gather(loader, n):
            xs, ys = [], []
            seen = 0
            for x_batch, y_batch in loader:
                xs.append(x_batch)
                ys.append(y_batch)
                seen += len(x_batch)
                if seen >= n:
                    break
            return torch.cat(xs)[:n], torch.cat(ys)[:n]

        train_x, train_y = gather(train_loader, max_samples)
        test_x, test_y = gather(test_loader, max_samples)
        return self.check_leakage(
            train_x.numpy(),
            train_y.numpy(),
            test_x.numpy(),
            test_y.numpy(),
            max_samples=max_samples,
            print_summary=print_summary,
        )

    def keras_callback(self):
        from gradglass.capture import GradGlassKerasCallback

        return GradGlassKerasCallback(run=self)

    def fit(self, X, y=None, X_val=None, y_val=None, eval_set=None, **fit_params):
        if not hasattr(self, "engine"):
            raise RuntimeError("Call run.watch(model) before run.fit().")
        if self.framework not in ("sklearn", "xgboost", "lightgbm"):
            raise RuntimeError(
                "run.fit() is intended for sklearn / XGBoost / LightGBM models. "
                "For PyTorch or Keras use a training loop with run.log()."
            )

        kw = dict(fit_params)
        # Only pass eval_set to frameworks that support it (XGBoost / LightGBM)
        if self.framework in ("xgboost", "lightgbm"):
            if eval_set is not None:
                kw.setdefault("eval_set", eval_set)
            elif X_val is not None and y_val is not None:
                kw.setdefault("eval_set", [(X_val, y_val)])

        t0 = time.time()
        try:
            self.model.fit(X, y, **kw)
        except TypeError as exc:
            if "callbacks" in str(exc) and "callbacks" in kw:
                # Older XGBoost / LightGBM sklearn API does not accept callbacks= in fit().
                # Strip it and retry; per-round metrics will be recovered from evals_result_.
                kw.pop("callbacks")
                self.model.fit(X, y, **kw)
            else:
                raise
        duration = time.time() - t0

        # For XGBoost / LightGBM: if no per-round metrics were written by a callback
        # (step is still 0), recover them from the model's evals_result.
        if self.framework in ("xgboost", "lightgbm") and self.step == 0:
            evals = None
            try:
                if hasattr(self.model, "evals_result_"):
                    evals = self.model.evals_result_
                elif callable(getattr(self.model, "evals_result", None)):
                    evals = self.model.evals_result()
            except Exception:
                pass
            if evals:
                rounds = max((len(vals) for ds_m in evals.values() for vals in ds_m.values()), default=0)
                for r in range(rounds):
                    self.step = r + 1
                    rm = {"step": self.step, "timestamp": time.time()}
                    for ds, dsm in evals.items():
                        for mn, vals in dsm.items():
                            rm[f"{ds}_{mn}".replace("-", "_")] = float(vals[r])
                    with open(self.metrics_file, "a") as f:
                        f.write(json.dumps(rm, default=json_default) + "\n")

        self.step += 1
        entry = {"step": self.step, "fit_duration_s": round(duration, 3)}
        estimator_type = getattr(self.model, "_estimator_type", "unknown")
        metric = "accuracy" if estimator_type == "classifier" else "r2"

        if hasattr(self.model, "score") and y is not None:
            try:
                entry[f"train_{metric}"] = round(float(self.model.score(X, y)), 5)
            except Exception:
                pass
        if X_val is not None and y_val is not None and hasattr(self.model, "score"):
            try:
                entry[f"val_{metric}"] = round(float(self.model.score(X_val, y_val)), 5)
            except Exception:
                pass

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry, default=json_default) + "\n")

        if hasattr(self.engine, "capture_post_fit"):
            self.engine.capture_post_fit(step=self.step)
        self.engine.save_checkpoint(step=self.step, tag="post_fit")

        X_for_eval = X_val if X_val is not None else X
        y_for_eval = y_val if y_val is not None else y
        if y_for_eval is not None and hasattr(self.model, "predict"):
            try:
                y_pred_vals = self.model.predict(X_for_eval)
                if hasattr(self.model, "predict_proba"):
                    try:
                        y_proba = self.model.predict_proba(X_for_eval)
                        self.engine.log_batch_predictions(
                            step=self.step, x=X_for_eval, y=y_for_eval, y_pred=y_proba, loss=None
                        )
                    except Exception:
                        self.engine.log_batch_predictions(
                            step=self.step, x=X_for_eval, y=y_for_eval, y_pred=y_pred_vals, loss=None
                        )
                else:
                    self.engine.log_batch_predictions(
                        step=self.step, x=X_for_eval, y=y_for_eval, y_pred=y_pred_vals, loss=None
                    )
            except Exception:
                pass
        self._write_runtime_state(event="fit", current_step=self.step)

        return self.model

    def xgboost_callback(self):

        from gradglass.capture import XGBoostGradGlassCallback

        return XGBoostGradGlassCallback(self).get()

    def lightgbm_callback(self):
        from gradglass.capture import LightGBMGradGlassCallback

        return LightGBMGradGlassCallback(self)

    def get_lr(self):
        if self.optimizer is None:
            return None
        if hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups[0].get("lr")
        if hasattr(self.optimizer, "learning_rate"):
            lr = self.optimizer.learning_rate
            if callable(lr):
                return float(lr(self.step))
            return float(lr)
        return None

    def __repr__(self):
        return f"<GradGlass.Run name='{self.name}' id='{self.run_id}' step={self.step}>"


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _safe_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
