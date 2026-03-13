from __future__ import annotations
import json
import time
import threading
from pathlib import Path
from typing import Any, Optional
from queue import Queue
import numpy as np


class CaptureEngine:
    def __init__(self, model, optimizer, framework, run_dir, config):
        self.model = model
        self.optimizer = optimizer
        self.framework = framework
        self.run_dir = run_dir
        self.config = config
        self.hooks = []
        self.activation_buffer = {}
        self.gradient_buffer = {}
        self.prediction_buffer = []
        self.write_queue = Queue()
        self.writer_thread = None
        self.running = True
        self.writer_thread = threading.Thread(target=self.background_writer, daemon=True)
        self.writer_thread.start()
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "gradients").mkdir(parents=True, exist_ok=True)
        (run_dir / "activations").mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    def extract_architecture(self):
        if self.framework == "pytorch":
            structure = self.extract_pytorch_architecture()
        elif self.framework == "tensorflow":
            structure = self.extract_tensorflow_architecture()
        else:
            structure = {"layers": [], "edges": []}
        with open(self.run_dir / "model_structure.json", "w") as f:
            json.dump(structure, f, indent=2)
        return structure

    def extract_pytorch_architecture(self):
        # ── collect every named sub-module ──────────────────────────────────
        raw: dict[str, dict] = {}
        root_type = type(self.model).__name__

        for name, module in self.model.named_modules():
            if name == "":
                continue
            layer_type = type(module).__name__
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = any(p.requires_grad for p in module.parameters(recurse=False))
            params_info: dict = {}
            for pname, p in module.named_parameters(recurse=False):
                params_info[pname] = list(p.shape)

            parts = name.split(".")
            depth = len(parts)
            parent_id = ".".join(parts[:-1]) if depth > 1 else "__root__"

            raw[name] = {
                "id": name,
                "type": layer_type,
                "params": params_info,
                "param_count": param_count,
                "trainable": trainable,
                "input_shape": None,
                "output_shape": None,
                "depth": depth,
                "parent": parent_id,
                "children": [],
                "is_container": False,
                "category": self._infer_category(layer_type),
            }

        # ── build children lists and mark containers ─────────────────────────
        for name, info in raw.items():
            pid = info["parent"]
            if pid == "__root__":
                continue
            if pid in raw:
                raw[pid]["children"].append(name)
                raw[pid]["is_container"] = True

        # ── build edge list (parent → child, top-level siblings chained) ────
        edges: list[list[str]] = []
        seen_edges: set[tuple] = set()
        top_level = [n for n, v in raw.items() if v["parent"] == "__root__"]

        def add_edge(src, dst):
            key = (src, dst)
            if key not in seen_edges:
                seen_edges.add(key)
                edges.append([src, dst])

        # parent → each direct child
        for name, info in raw.items():
            for child in info["children"]:
                add_edge(name, child)

        # chain top-level modules sequentially (data flows L→R)
        for i in range(len(top_level) - 1):
            add_edge(top_level[i], top_level[i + 1])

        layers = list(raw.values())
        return {
            "layers": layers,
            "edges": edges,
            "root_type": root_type,
            "top_level": top_level,
        }

    # ── category inference ────────────────────────────────────────────────────
    @staticmethod
    def _infer_category(layer_type: str) -> str:
        lt = layer_type.lower()
        if any(k in lt for k in ("conv", "linear", "embedding", "attention", "transformer",
                                  "lstm", "gru", "rnn", "encoder", "decoder", "res2net",
                                  "multihead", "selfattn", "crossattn", "pool", "ecapa",
                                  "speaker", "diarization", "sequential", "modulelist",
                                  "moduledict", "feedforward", "mlp", "head")):
            return "model"
        if any(k in lt for k in ("batchnorm", "layernorm", "groupnorm", "instancenorm",
                                  "dropout", "relu", "gelu", "silu", "activation",
                                  "softmax", "sigmoid", "tanh", "norm")):
            return "model"
        if any(k in lt for k in ("loss", "criterion", "crossentropy", "bce", "mse",
                                  "nll", "kl", "contrastive", "triplet", "focal")):
            return "training"
        if any(k in lt for k in ("metric", "accuracy", "f1", "auc", "eval", "score")):
            return "eval"
        if any(k in lt for k in ("dataset", "dataloader", "transform", "augment",
                                  "preprocess", "feature", "tokenizer", "collate")):
            return "data"
        return "model"

    def extract_tensorflow_architecture(self):
        layers = []
        edges = []
        try:
            for i, layer in enumerate(self.model.layers):
                layer_type = type(layer).__name__
                param_count = layer.count_params() if hasattr(layer, "count_params") else 0
                trainable = getattr(layer, "trainable", True)
                config = {}
                try:
                    config = layer.get_config()
                except Exception:
                    pass
                input_shape = None
                output_shape = None
                try:
                    input_shape = layer.input_shape if hasattr(layer, "input_shape") else None
                    output_shape = layer.output_shape if hasattr(layer, "output_shape") else None
                except (AttributeError, RuntimeError):
                    pass
                layer_info = {
                    "id": layer.name,
                    "type": layer_type,
                    "params": config,
                    "param_count": param_count,
                    "trainable": trainable,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                }
                layers.append(layer_info)
                if i > 0:
                    edges.append([self.model.layers[i - 1].name, layer.name])
        except Exception:
            pass
        return {"layers": layers, "edges": edges}

    def attach_hooks(self):
        if self.framework == "pytorch":
            self.attach_pytorch_hooks()
        elif self.framework == "tensorflow":
            self.attach_tensorflow_hooks()

    def attach_tensorflow_hooks(self):
        import tensorflow as tf

        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is not None:
            _original_apply = optimizer.apply_gradients

            def _patched_apply(grads_and_vars, *args, **kwargs):
                for grad, var in grads_and_vars:
                    if grad is None:
                        continue
                    try:
                        g_np = grad.numpy()
                        summary = {
                            "mean": float(np.mean(g_np)),
                            "var": float(np.var(g_np)),
                            "max": float(np.max(np.abs(g_np))),
                            "norm": float(np.linalg.norm(g_np)),
                            "min": float(np.min(g_np)),
                        }
                        if var.name not in self.gradient_buffer:
                            self.gradient_buffer[var.name] = []
                        self.gradient_buffer[var.name].append(summary)
                    except Exception:
                        pass
                return _original_apply(grads_and_vars, *args, **kwargs)

            optimizer.apply_gradients = _patched_apply
            self._tf_optimizer = optimizer
            self._tf_original_apply = _original_apply

        self._tf_last_input = None
        _original_model_call = self.model.__call__

        def _patched_call(inputs, *args, **kwargs):
            try:
                self._tf_last_input = inputs
            except Exception:
                pass
            return _original_model_call(inputs, *args, **kwargs)

        self.model.__call__ = _patched_call
        self._tf_original_call = _original_model_call

        self._tf_activation_extractors = {}
        layers_config = self.config.get("layers", "trainable")
        if hasattr(self.model, "inputs") and self.model.inputs is not None:
            for layer in self.model.layers:
                if not layer.trainable_weights:
                    continue
                if layers_config == "trainable" and not layer.trainable:
                    continue
                if isinstance(layers_config, list) and layer.name not in layers_config:
                    continue
                try:
                    extractor = tf.keras.Model(inputs=self.model.inputs, outputs=layer.output)
                    self._tf_activation_extractors[layer.name] = extractor
                except Exception:
                    pass

    def attach_pytorch_hooks(self):
        import torch

        layers_config = self.config.get("layers", "trainable")
        act_config = self.config.get("activations", "auto")
        sample_batches = self.config.get("sample_batches", 2)
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if layers_config == "trainable":
                has_params = any((p.requires_grad for p in module.parameters(recurse=False)))
                if not has_params:
                    continue
            elif isinstance(layers_config, list):
                if name not in layers_config:
                    continue
            if act_config != "off":
                hook = module.register_forward_hook(self.make_forward_hook(name, sample_batches))
                self.hooks.append(hook)
            if self.config.get("gradients", "summary") != "off":
                for pname, param in module.named_parameters(recurse=False):
                    if param.requires_grad:
                        full_name = f"{name}.{pname}"
                        hook = param.register_hook(self.make_grad_hook(full_name))
                        self.hooks.append(hook)

    def make_forward_hook(self, layer_name, sample_batches):
        batch_count = [0]

        def hook(module, input, output):
            try:
                if hasattr(output, "shape"):
                    shape = list(output.shape)
                    self.update_layer_shape(layer_name, input, shape)
            except Exception:
                pass
            if batch_count[0] < sample_batches:
                try:
                    if hasattr(output, "detach"):
                        act_data = output.detach().cpu().numpy()
                    elif hasattr(output, "numpy"):
                        act_data = output.numpy()
                    else:
                        act_data = np.array(output)
                    if layer_name not in self.activation_buffer:
                        self.activation_buffer[layer_name] = []
                    self.activation_buffer[layer_name].append(act_data)
                    batch_count[0] += 1
                except Exception:
                    pass

        return hook

    def make_grad_hook(self, param_name):

        def hook(grad):
            try:
                grad_np = grad.detach().cpu().numpy()
                summary = {
                    "mean": float(np.mean(grad_np)),
                    "var": float(np.var(grad_np)),
                    "max": float(np.max(np.abs(grad_np))),
                    "norm": float(np.linalg.norm(grad_np)),
                    "min": float(np.min(grad_np)),
                }
                if param_name not in self.gradient_buffer:
                    self.gradient_buffer[param_name] = []
                self.gradient_buffer[param_name].append(summary)
            except Exception:
                pass

        return hook

    def update_layer_shape(self, layer_name, input_data, output_shape):
        arch_path = self.run_dir / "model_structure.json"
        if not arch_path.exists():
            return
        try:
            with open(arch_path) as f:
                structure = json.load(f)
            for layer in structure["layers"]:
                if layer["id"] == layer_name:
                    layer["output_shape"] = output_shape
                    if isinstance(input_data, tuple) and len(input_data) > 0:
                        first_input = input_data[0]
                        if hasattr(first_input, "shape"):
                            layer["input_shape"] = list(first_input.shape)
                    break
            with open(arch_path, "w") as f:
                json.dump(structure, f, indent=2)
        except Exception:
            pass

    def save_checkpoint(self, step, tag=None):
        ckpt_dir = self.run_dir / "checkpoints"
        filename = f"step_{step}.npz"
        weights = {}
        if self.framework == "pytorch":
            for name, param in self.model.named_parameters():
                weights[name] = param.detach().cpu().numpy()
        elif self.framework == "tensorflow":
            for var in self.model.trainable_variables:
                weights[var.name] = var.numpy()
        self.write_queue.put(("checkpoint", ckpt_dir / filename, weights))
        meta = {
            "step": step,
            "tag": tag,
            "timestamp": time.time(),
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_params": sum((w.size for w in weights.values())),
        }
        meta_path = ckpt_dir / f"step_{step}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        self.flush_activations(step)

    def capture_gradients(self, step):
        if not self.gradient_buffer:
            return
        grad_dir = self.run_dir / "gradients"
        summaries = {}
        for param_name, grad_list in self.gradient_buffer.items():
            if not grad_list:
                continue
            summaries[param_name] = {
                "mean": float(np.mean([g["mean"] for g in grad_list])),
                "var": float(np.mean([g["var"] for g in grad_list])),
                "max": float(np.max([g["max"] for g in grad_list])),
                "norm": float(np.mean([g["norm"] for g in grad_list])),
            }
        prev_path = self.find_previous_gradient_summary(step)
        if prev_path is not None:
            try:
                with open(prev_path) as f:
                    prev_summaries = json.load(f)
                for param_name in summaries:
                    if param_name in prev_summaries:
                        prev = prev_summaries[param_name]
                        curr = summaries[param_name]
                        if prev["var"] > 0 and curr["var"] > 0:
                            kl = (
                                np.log(curr["var"] / prev["var"])
                                + (prev["var"] + (prev["mean"] - curr["mean"]) ** 2) / (2 * curr["var"])
                                - 0.5
                            )
                            summaries[param_name]["kl_div"] = float(abs(kl))
            except Exception:
                pass
        summary_path = grad_dir / f"summaries_step_{step}.json"
        with open(summary_path, "w") as f:
            json.dump(summaries, f, indent=2)
        self.gradient_buffer.clear()

    def find_previous_gradient_summary(self, current_step):
        grad_dir = self.run_dir / "gradients"
        if not grad_dir.exists():
            return None
        summary_files = sorted(grad_dir.glob("summaries_step_*.json"))
        for f in reversed(summary_files):
            try:
                step = int(f.stem.split("_")[-1])
                if step < current_step:
                    return f
            except ValueError:
                continue
        return None

    def flush_activations(self, step):
        act_dir = self.run_dir / "activations"

        # TensorFlow path: extract activations by running sub-models on cached input
        if self.framework == "tensorflow":
            if hasattr(self, "_tf_activation_extractors") and self._tf_last_input is not None:
                for layer_name, extractor in self._tf_activation_extractors.items():
                    try:
                        act_output = extractor(self._tf_last_input, training=False)
                        act_data = act_output.numpy()
                        safe_name = layer_name.replace(".", "_")
                        filepath = act_dir / f"{safe_name}_step_{step}.npy"
                        np.save(filepath, act_data)
                    except Exception:
                        pass
            return

        # PyTorch path: flush activation buffer populated by forward hooks
        for layer_name, act_list in self.activation_buffer.items():
            if not act_list:
                continue
            safe_name = layer_name.replace(".", "_")
            filepath = act_dir / f"{safe_name}_step_{step}.npy"
            try:
                combined = np.concatenate(act_list, axis=0)
                np.save(filepath, combined)
            except Exception:
                stats = {
                    "mean": float(np.mean([np.mean(a) for a in act_list])),
                    "var": float(np.mean([np.var(a) for a in act_list])),
                    "sparsity": float(np.mean([np.sum(np.abs(a) < 1e-06) / a.size for a in act_list])),
                    "shape": list(act_list[0].shape) if act_list else [],
                }
                stats_path = act_dir / f"{safe_name}_step_{step}_stats.json"
                with open(stats_path, "w") as f:
                    json.dump(stats, f, indent=2)
        self.activation_buffer.clear()

    @staticmethod
    def detect_leakage(x, y):
        pass

    def log_batch_predictions(self, step, x, y, y_pred, loss):
        pred_dir = self.run_dir / "predictions"
        try:

            def to_np(t):
                if t is None:
                    return None
                if hasattr(t, "detach"):
                    return t.detach().cpu().numpy()
                if hasattr(t, "numpy"):
                    return t.numpy()
                return np.array(t)

            y_np = to_np(y)
            y_pred_np = to_np(y_pred)
            record = {"step": step, "timestamp": time.time()}
            if y_np is not None:
                record["y_true"] = y_np[:256].tolist() if len(y_np) > 256 else y_np.tolist()
            if y_pred_np is not None:
                if y_pred_np.ndim > 1:
                    pred_classes = np.argmax(y_pred_np, axis=-1)
                    confidence = np.max(y_pred_np, axis=-1)
                    record["y_pred"] = pred_classes[:256].tolist()
                    record["confidence"] = confidence[:256].tolist()
                    record["logits_sample"] = y_pred_np[:16].tolist()
                else:
                    record["y_pred"] = y_pred_np[:256].tolist()
            if loss is not None:
                record["loss"] = float(loss) if np.isscalar(loss) else float(to_np(loss))
            filepath = pred_dir / f"probe_step_{step}.json"
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception:
            pass

    def background_writer(self):
        from queue import Empty

        while self.running:
            try:
                item = self.write_queue.get(timeout=1.0)
            except Empty:
                continue
            try:
                if item is None:
                    break
                (kind, path, data) = item
                if kind == "checkpoint":
                    np.savez_compressed(str(path), **data)
                elif kind == "activation":
                    np.save(str(path), data)
            except Exception:
                pass
            finally:
                self.write_queue.task_done()

    def flush_writes(self, timeout=30.0):
        self.write_queue.join()

    def cleanup(self):
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()
        self.running = False
        self.write_queue.put(None)
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5.0)
        # Restore TensorFlow patches if they were applied
        if self.framework == "tensorflow":
            if hasattr(self, "_tf_optimizer") and hasattr(self, "_tf_original_apply"):
                try:
                    self._tf_optimizer.apply_gradients = self._tf_original_apply
                except Exception:
                    pass
            if hasattr(self, "_tf_original_call"):
                try:
                    self.model.__call__ = self._tf_original_call
                except Exception:
                    pass


def safe_json(v):
    """Convert a value to a JSON-serialisable type."""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _safe_positive_int(value):
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class SklearnCaptureAdapter:
    """Capture adapter for scikit-learn estimators.

    Replaces ``CaptureEngine`` for sklearn-based models.  Gradient hooks are
    not applicable here, but GradGlass still captures:

    * Hyperparameters / pipeline steps  (on ``extract_architecture``)
    * Feature importances, coefficients, training curves  (on ``capture_post_fit``)
    * Serialised model checkpoint via joblib  (on ``save_checkpoint``)
    """

    def __init__(self, model, run_dir, config):
        self.model = model
        self.run_dir = run_dir
        self.config = config
        for subdir in ("checkpoints", "gradients", "activations", "predictions", "slices"):
            (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Architecture
    # ------------------------------------------------------------------

    def extract_architecture(self):
        model = self.model
        model_type = type(model).__name__
        module = type(model).__module__
        estimator_type = getattr(model, "_estimator_type", "unknown")

        # Determine actual framework
        if "xgboost" in module:
            framework = "xgboost"
        elif "lightgbm" in module:
            framework = "lightgbm"
        else:
            framework = "sklearn"

        # Detect tree/boosting-based models
        tree_kw = ["forest", "boost", "tree", "gradient", "ada", "bagging", "extra", "xgb", "lgb", "random"]
        is_boosting = framework in ("xgboost", "lightgbm") or any(kw in model_type.lower() for kw in tree_kw)

        params: dict = {}
        try:
            params = {k: safe_json(v) for k, v in model.get_params(deep=True).items()}
        except Exception:
            pass

        # Pipeline step breakdown
        pipeline_steps = []
        if hasattr(model, "steps"):
            for sname, step in model.steps:
                step_params: dict = {}
                try:
                    step_params = {k: safe_json(v) for k, v in step.get_params(deep=False).items()}
                except Exception:
                    pass
                pipeline_steps.append(
                    {
                        "name": sname,
                        "type": type(step).__name__,
                        "module": type(step).__module__,
                        "estimator_type": getattr(step, "_estimator_type", "unknown"),
                        "params": step_params,
                    }
                )

        structure = {
            "framework": framework,
            "is_boosting": is_boosting,
            "layers": [
                {
                    "id": "estimator",
                    "type": model_type,
                    "module": module,
                    "estimator_type": estimator_type,
                    "params": params,
                    "param_count": 0,
                    "trainable": True,
                    "input_shape": None,
                    "output_shape": None,
                }
            ],
            "edges": [],
            "pipeline_steps": pipeline_steps,
        }
        with open(self.run_dir / "model_structure.json", "w") as f:
            json.dump(structure, f, indent=2)
        return structure

    # ------------------------------------------------------------------
    # Post-fit diagnostics
    # ------------------------------------------------------------------

    def capture_post_fit(self, step: int = 1) -> dict:
        """Capture feature importances, coefficients and training diagnostics."""
        model = self.model
        # For pipelines, also inspect the final estimator
        final = getattr(model, "named_steps", {})
        final_estimator = list(final.values())[-1] if final else model

        summary: dict = {}

        for obj in (model, final_estimator):
            if hasattr(obj, "feature_importances_") and "feature_importances" not in summary:
                summary["feature_importances"] = obj.feature_importances_.tolist()
            if hasattr(obj, "coef_") and "coef" not in summary:
                summary["coef"] = np.array(obj.coef_).tolist()
            if hasattr(obj, "intercept_") and "intercept" not in summary:
                summary["intercept"] = np.array(obj.intercept_).tolist()
            if hasattr(obj, "n_iter_") and "n_iter" not in summary:
                n_iter = obj.n_iter_
                summary["n_iter"] = int(n_iter) if np.isscalar(n_iter) else np.array(n_iter).tolist()
            if hasattr(obj, "loss_curve_") and "loss_curve" not in summary:
                summary["loss_curve"] = [float(x) for x in obj.loss_curve_]
            if hasattr(obj, "best_loss_") and "best_loss" not in summary:
                summary["best_loss"] = float(obj.best_loss_)
            if hasattr(obj, "inertia_") and "inertia" not in summary:
                summary["inertia"] = float(obj.inertia_)
            if hasattr(obj, "explained_variance_ratio_") and "explained_variance_ratio" not in summary:
                summary["explained_variance_ratio"] = obj.explained_variance_ratio_.tolist()
            if hasattr(obj, "singular_values_") and "singular_values" not in summary:
                summary["singular_values"] = obj.singular_values_.tolist()
            if hasattr(obj, "components_") and "n_components" not in summary:
                summary["n_components"] = obj.components_.shape[0]
            if hasattr(obj, "cluster_centers_") and "n_clusters" not in summary:
                summary["n_clusters"] = len(obj.cluster_centers_)
            if hasattr(obj, "labels_") and "n_clusters_found" not in summary:
                summary["n_clusters_found"] = int(np.max(obj.labels_) + 1)
            if hasattr(obj, "oob_score_") and "oob_score" not in summary:
                summary["oob_score"] = float(obj.oob_score_)

        if summary:
            # Use a separate filename so the artifact store's gradient-summary
            # reader (which expects per-layer gradient stats) is not confused.
            path = self.run_dir / "gradients" / f"sklearn_diagnostics_step_{step}.json"
            with open(path, "w") as f:
                json.dump(summary, f, indent=2)

        # ── Update model_structure.json with post-fit diagnostics ──────────
        arch_path = self.run_dir / "model_structure.json"
        if arch_path.exists():
            try:
                with open(arch_path) as f:
                    structure = json.load(f)

                if "feature_importances" in summary:
                    structure["feature_importances"] = summary["feature_importances"]

                # XGBoost booster tree dump
                for obj in (model, final_estimator):
                    if hasattr(obj, "get_booster"):
                        try:
                            booster = obj.get_booster()
                            dump = booster.get_dump()
                            structure["trees_info"] = {"num_trees": len(dump), "tree_dump_sample": dump[:3]}
                            # Per-round evals result
                            try:
                                er = obj.evals_result()
                                if er:
                                    structure["evals_result"] = {
                                        ds: {m: list(vals) for m, vals in dsm.items()} for ds, dsm in er.items()
                                    }
                            except Exception:
                                pass
                        except Exception:
                            pass
                        break
                    elif hasattr(obj, "estimators_"):
                        try:
                            ests = obj.estimators_
                            if isinstance(ests, np.ndarray):
                                n = int(ests.size)
                            elif isinstance(ests, list):
                                n = sum(len(s) if hasattr(s, "__len__") else 1 for s in ests)
                            else:
                                _n = 0
                            if n:
                                structure.setdefault("trees_info", {})["num_trees"] = n
                        except Exception:
                            pass
                        break

                with open(arch_path, "w") as f:
                    json.dump(structure, f, indent=2)
            except Exception:
                pass

        return summary

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, step: int = 1, tag=None):
        ckpt_dir = self.run_dir / "checkpoints"
        filepath = ckpt_dir / f"step_{step}.pkl"
        try:
            import joblib

            joblib.dump(self.model, str(filepath))
            fmt = "joblib"
        except ImportError:
            import pickle

            with open(filepath, "wb") as fh:
                pickle.dump(self.model, fh)
            fmt = "pickle"
        meta = {
            "step": step,
            "tag": tag,
            "timestamp": time.time(),
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_type": type(self.model).__name__,
            "format": fmt,
            "num_params": 0,
        }
        with open(ckpt_dir / f"step_{step}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # No-op stubs (keep interface compatible with CaptureEngine)
    # ------------------------------------------------------------------

    def attach_hooks(self):
        pass

    def capture_gradients(self, step):
        pass

    def flush_activations(self, step):
        pass

    def log_batch_predictions(self, step, x, y, y_pred, loss):
        pred_dir = self.run_dir / "predictions"
        try:

            def to_np(t):
                if t is None:
                    return None
                if hasattr(t, "detach"):
                    return t.detach().cpu().numpy()
                if hasattr(t, "numpy"):
                    return t.numpy()
                return np.array(t)

            y_np = to_np(y)
            y_pred_np = to_np(y_pred)
            record: dict = {"step": step, "timestamp": time.time()}
            if y_np is not None:
                record["y_true"] = y_np[:1000].tolist() if len(y_np) > 1000 else y_np.tolist()
            if y_pred_np is not None:
                if y_pred_np.ndim > 1:
                    pred_classes = np.argmax(y_pred_np, axis=-1)
                    record["y_pred"] = (
                        pred_classes[:1000].tolist() if len(pred_classes) > 1000 else pred_classes.tolist()
                    )
                else:
                    record["y_pred"] = y_pred_np[:1000].tolist() if len(y_pred_np) > 1000 else y_pred_np.tolist()
            if loss is not None:
                record["loss"] = float(loss)
            filepath = pred_dir / f"probe_step_{step}.json"
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2, default=str)
        except Exception:
            pass

    def cleanup(self):
        pass

    def flush_writes(self):
        pass


class XGBoostGradGlassCallback:
    """XGBoost ``TrainingCallback`` that streams per-round eval metrics to a Run.

    Usage::

        cb = run.xgboost_callback()
        booster = xgb.train(params, dtrain, evals=[(dval, 'val')],
                            callbacks=[cb])

    Works with both the functional ``xgb.train()`` API and the
    ``XGBClassifier / XGBRegressor`` sklearn-compatible ``.fit()`` API.
    """

    def __init__(self, run):
        self.run_ref = run
        try:
            import xgboost as xgb

            run_ref = run

            class CB(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    metrics: dict = {}
                    for dataset, dataset_metrics in evals_log.items():
                        for metric_name, values in dataset_metrics.items():
                            if values:
                                key = f"{dataset}_{metric_name}".replace("-", "_")
                                metrics[key] = float(values[-1])
                    if metrics:
                        run_ref.log(**metrics)
                    return False  # do not stop training

            self.cb = CB()
        except (ImportError, AttributeError):
            self.cb = self

    def __call__(self, env):
        """Legacy XGBoost callback interface (pre-1.3)."""
        metrics: dict = {}
        for k, v in getattr(env, "evaluation_result_list", []):
            metrics[k.replace("-", "_").replace(" ", "_")] = float(v)
        if metrics:
            self.run_ref.log(**metrics)

    def get(self):
        """Return the XGBoost callback object to pass to ``callbacks=``."""
        return self.cb


class LightGBMGradGlassCallback:
    """LightGBM callback that streams per-round eval metrics to a Run.

    Usage::

        cb = run.lightgbm_callback()
        gbm = lgb.train(params, train_data, valid_sets=[val_data],
                        callbacks=[cb])

    Also works with ``LGBMClassifier / LGBMRegressor`` ``.fit()``::

        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[run.lightgbm_callback()])
    """

    # LightGBM inspects these attributes on the callback
    order: int = 10
    before_iteration: bool = False

    def __init__(self, run):
        self.run_ref = run

    def __call__(self, env):
        metrics: dict = {}
        for data_name, eval_name, result, _ in getattr(env, "evaluation_result_list", []):
            key = f"{data_name}_{eval_name}".replace("-", "_").replace(" ", "_")
            metrics[key] = float(result)
        if metrics:
            self.run_ref.log(**metrics)


try:
    import tensorflow as _tf_for_keras_cb

    _KerasCallbackBase = _tf_for_keras_cb.keras.callbacks.Callback
except ImportError:
    _KerasCallbackBase = object


class GradGlassKerasCallback(_KerasCallbackBase):
    def __init__(self, run):
        if _KerasCallbackBase is not object:
            super().__init__()
        self.run = run

    def on_train_begin(self, logs=None):
        params = getattr(self, "params", {}) or {}
        total_steps = None

        epochs = _safe_positive_int(params.get("epochs"))
        steps_per_epoch = _safe_positive_int(params.get("steps"))
        if steps_per_epoch is None:
            steps_per_epoch = _safe_positive_int(params.get("steps_per_epoch"))

        if epochs and steps_per_epoch:
            total_steps = epochs * steps_per_epoch

        self.run._write_runtime_state(
            status="running",
            event="keras_train_begin",
            current_step=self.run.step,
            total_steps=total_steps,
            fatal_exception=None,
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.run.log(**logs)
        self.run.checkpoint(tag=f"epoch_{epoch}")

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if "loss" in logs:
            self.run.step += 1
            entry = {"step": self.run.step, "timestamp": time.time(), **logs}
            with open(self.run.metrics_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        self.run._write_runtime_state(event="keras_batch_end", current_step=self.run.step)
