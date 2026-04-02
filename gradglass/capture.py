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
        self.probe_examples = max(1, int(config.get("probe_examples", 16) or 16))
        self.saliency_mode = config.get("saliency", "auto")
        self.activation_capture_counts = {}
        self.latest_probe_activations = {}
        self.cached_probe_activations = {}
        self._capture_checkpoint_activations = True
        self._capture_probe_activations = False
        self.writer_thread = threading.Thread(target=self.background_writer, daemon=True)
        self.writer_thread.start()
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "gradients").mkdir(parents=True, exist_ok=True)
        (run_dir / "activations").mkdir(parents=True, exist_ok=True)
        (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (run_dir / "probes").mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _safe_key(name):
        return "".join((ch if ch.isalnum() else "_" for ch in str(name)))

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            if len(value) == 1:
                return CaptureEngine._to_numpy(value[0])
            try:
                arr = np.asarray(value)
                if arr.dtype != object:
                    return arr
            except Exception:
                pass
            return CaptureEngine._to_numpy(value[0])
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        if hasattr(value, "numpy"):
            return value.numpy()
        try:
            return np.asarray(value)
        except Exception:
            return None

    @staticmethod
    def _slice_batch(value, limit):
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(CaptureEngine._slice_batch(v, limit) for v in value)
        if isinstance(value, list):
            return [CaptureEngine._slice_batch(v, limit) for v in value]
        try:
            return value[:limit]
        except Exception:
            return value

    @staticmethod
    def _extract_primary_value(value):
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return CaptureEngine._extract_primary_value(value[0])
        return value

    def _extract_primary_input_array(self, value):
        primary = self._extract_primary_value(value)
        arr = self._to_numpy(primary)
        if arr is None:
            return None
        if getattr(arr, "ndim", 0) > 0:
            return arr[: self.probe_examples]
        return arr

    @staticmethod
    def _infer_input_modality(x_np):
        if x_np is None:
            return None
        if x_np.ndim >= 4:
            return "vision"
        if x_np.ndim == 3:
            return "sequence"
        if x_np.ndim == 2:
            return "structured_data"
        return "scalar_or_label"

    @staticmethod
    def _looks_like_class_scores(y_np, y_pred_np):
        if y_np is None or y_pred_np is None or getattr(y_pred_np, "ndim", 0) != 2:
            return False
        if getattr(y_np, "ndim", 0) > 1:
            return False
        if y_pred_np.shape[0] != y_np.shape[0] or y_pred_np.shape[1] <= 1:
            return False
        try:
            float_arr = y_np.astype(float)
        except (TypeError, ValueError, AttributeError):
            return False
        if np.any(~np.isfinite(float_arr)):
            return False
        return bool(np.all(np.isclose(float_arr, np.round(float_arr))))

    @staticmethod
    def _confidence_from_scores(scores):
        scores = scores.astype(float)
        if np.all(scores >= 0.0):
            row_sums = scores.sum(axis=1)
            if np.all(np.isclose(row_sums, 1.0, atol=1e-3)):
                return np.max(scores, axis=1)
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        return np.max(probs, axis=1)

    @staticmethod
    def _reduce_prediction_array(y_pred_np):
        if y_pred_np is None:
            return None, None, None
        if getattr(y_pred_np, "ndim", 0) == 2 and y_pred_np.shape[1] > 1:
            pred = np.argmax(y_pred_np, axis=-1)
            confidence = CaptureEngine._confidence_from_scores(y_pred_np)
            return pred, confidence, y_pred_np
        return y_pred_np, None, None

    def _reset_activation_capture_window(self):
        self.activation_buffer.clear()
        self.activation_capture_counts.clear()

    def _reset_probe_window(self):
        self.latest_probe_activations.clear()
        self._reset_activation_capture_window()

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
        def hook(module, input, output):
            try:
                if hasattr(output, "shape"):
                    shape = list(output.shape)
                    self.update_layer_shape(layer_name, input, shape)
            except Exception:
                pass
            act_data = self._to_numpy(output)
            if act_data is None:
                return
            if self._capture_checkpoint_activations:
                batch_count = self.activation_capture_counts.get(layer_name, 0)
                if batch_count < sample_batches:
                    self.activation_buffer.setdefault(layer_name, []).append(act_data)
                    self.activation_capture_counts[layer_name] = batch_count + 1
            if self._capture_probe_activations:
                try:
                    probe_slice = act_data[: self.probe_examples] if getattr(act_data, "ndim", 0) > 0 else act_data
                    self.latest_probe_activations[layer_name] = np.asarray(probe_slice)
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

    def _call_model(self, model_input):
        if isinstance(model_input, tuple):
            return self.model(*model_input)
        if isinstance(model_input, list):
            return self.model(*model_input)
        return self.model(model_input)

    def _capture_pytorch_probe_bundle(self, x):
        import torch

        probe_input = self._slice_batch(x, self.probe_examples)
        primary_input = self._extract_primary_value(probe_input)

        bundle = {
            "activation_arrays": {},
            "prediction_array": None,
            "confidence_array": None,
            "logits_array": None,
            "saliency_array": None,
            "saliency_reason": "Not captured for this run.",
            "saliency_kind": None,
        }

        if probe_input is None:
            bundle["saliency_reason"] = "No input batch was available for probe capture."
            return bundle

        was_training = bool(getattr(self.model, "training", False))
        self.latest_probe_activations.clear()
        prev_checkpoint_capture = self._capture_checkpoint_activations
        self._capture_checkpoint_activations = False
        self._capture_probe_activations = True

        try:
            self.model.eval()
            with torch.enable_grad():
                outputs = self._call_model(probe_input)
                output_np = self._to_numpy(outputs)
                pred_array, confidence_array, logits_array = self._reduce_prediction_array(output_np)
                bundle["prediction_array"] = pred_array
                bundle["confidence_array"] = confidence_array
                bundle["logits_array"] = logits_array
                bundle["activation_arrays"] = {
                    layer: np.asarray(values)
                    for layer, values in self.latest_probe_activations.items()
                }

                saliency_disabled = str(self.saliency_mode).lower() == "off"
                if saliency_disabled:
                    bundle["saliency_reason"] = "Saliency capture was disabled for this run."
                elif primary_input is None or not torch.is_tensor(primary_input):
                    bundle["saliency_reason"] = "Saliency capture currently supports a single tensor input."
                elif not primary_input.is_floating_point():
                    bundle["saliency_reason"] = "Saliency capture is unavailable for non-floating-point inputs."
                elif primary_input.ndim not in (2, 4):
                    bundle["saliency_reason"] = "Saliency capture currently supports structured vectors and vision tensors."
                elif not torch.is_tensor(outputs):
                    bundle["saliency_reason"] = "Model outputs could not be converted into a saliency target."
                else:
                    grad_input = probe_input
                    if isinstance(probe_input, tuple) or isinstance(probe_input, list):
                        bundle["saliency_reason"] = "Saliency capture currently supports a single tensor input."
                    else:
                        grad_input = probe_input.detach().clone()
                        grad_input.requires_grad_(True)
                        self.latest_probe_activations.clear()
                        outputs = self._call_model(grad_input)
                        if outputs.ndim == 1:
                            selected = outputs.sum()
                            bundle["prediction_array"] = self._to_numpy(outputs)
                        elif outputs.ndim >= 2:
                            flat_outputs = outputs.reshape(outputs.shape[0], -1)
                            indices = flat_outputs.argmax(dim=1, keepdim=True)
                            selected = flat_outputs.gather(1, indices).sum()
                            bundle["prediction_array"] = indices.detach().cpu().numpy().reshape(-1)
                            bundle["confidence_array"] = self._confidence_from_scores(flat_outputs.detach().cpu().numpy())
                            bundle["logits_array"] = flat_outputs.detach().cpu().numpy()
                        else:
                            selected = outputs.sum()

                        try:
                            self.model.zero_grad(set_to_none=True)
                        except TypeError:
                            self.model.zero_grad()
                        selected.backward()
                        bundle["saliency_array"] = grad_input.grad.detach().abs().cpu().numpy()
                        bundle["saliency_kind"] = "vision" if grad_input.ndim == 4 else "structured"
                        bundle["saliency_reason"] = None
                        bundle["activation_arrays"] = {
                            layer: np.asarray(values)
                            for layer, values in self.latest_probe_activations.items()
                        }
                        try:
                            self.model.zero_grad(set_to_none=True)
                        except TypeError:
                            self.model.zero_grad()
        except Exception as exc:
            bundle["saliency_reason"] = f"Probe capture failed: {exc}"
        finally:
            self._capture_probe_activations = False
            self._capture_checkpoint_activations = prev_checkpoint_capture
            if was_training:
                self.model.train()
            self.latest_probe_activations.clear()

        return bundle

    def _capture_tensorflow_probe_bundle(self, x):
        bundle = {
            "activation_arrays": {},
            "prediction_array": None,
            "confidence_array": None,
            "logits_array": None,
            "saliency_array": None,
            "saliency_reason": "Saliency capture is currently unavailable for TensorFlow runs.",
            "saliency_kind": None,
        }

        if not hasattr(self, "_tf_activation_extractors"):
            return bundle

        probe_input = self._slice_batch(x, self.probe_examples)
        if probe_input is None:
            bundle["saliency_reason"] = "No input batch was available for probe capture."
            return bundle

        try:
            outputs = self.model(probe_input, training=False)
            output_np = self._to_numpy(outputs)
            pred_array, confidence_array, logits_array = self._reduce_prediction_array(output_np)
            bundle["prediction_array"] = pred_array
            bundle["confidence_array"] = confidence_array
            bundle["logits_array"] = logits_array
        except Exception:
            pass

        for layer_name, extractor in getattr(self, "_tf_activation_extractors", {}).items():
            try:
                act_output = extractor(probe_input, training=False)
                act_np = self._to_numpy(act_output)
                if act_np is not None:
                    bundle["activation_arrays"][layer_name] = act_np
            except Exception:
                continue

        return bundle

    def _write_probe_bundle(
        self,
        *,
        step,
        timestamp,
        primary_input,
        y_true_array,
        prediction_array,
        confidence_array,
        logits_array,
        activation_arrays,
        saliency_array,
        saliency_kind,
        saliency_reason,
    ):
        probe_dir = self.run_dir / "probes"
        data_path = probe_dir / f"probe_step_{step}.npz"
        meta_path = probe_dir / f"probe_step_{step}.json"

        tensors = {}
        activation_meta = []

        input_np = self._to_numpy(primary_input)
        if input_np is not None:
            if getattr(input_np, "ndim", 0) > 0:
                input_np = input_np[: self.probe_examples]
            tensors["input"] = np.asarray(input_np)

        if y_true_array is not None:
            tensors["targets"] = np.asarray(y_true_array)
        if prediction_array is not None:
            tensors["predictions"] = np.asarray(prediction_array)
        if confidence_array is not None:
            tensors["confidence"] = np.asarray(confidence_array)
        if logits_array is not None:
            tensors["logits"] = np.asarray(logits_array)
        if saliency_array is not None:
            tensors["saliency"] = np.asarray(saliency_array)

        for layer_name, values in sorted((activation_arrays or {}).items()):
            arr = np.asarray(values)
            key = f"activation__{self._safe_key(layer_name)}"
            tensors[key] = arr
            activation_meta.append({"layer": layer_name, "key": key, "shape": list(arr.shape)})

        np.savez_compressed(str(data_path), **tensors)

        input_modality = self._infer_input_modality(tensors.get("input"))
        meta = {
            "step": step,
            "timestamp": float(timestamp),
            "probe_examples": int(tensors.get("input").shape[0]) if isinstance(tensors.get("input"), np.ndarray) and tensors["input"].ndim > 0 else 0,
            "input_modality": input_modality,
            "input_shape": list(tensors["input"].shape) if isinstance(tensors.get("input"), np.ndarray) else None,
            "target_shape": list(tensors["targets"].shape) if isinstance(tensors.get("targets"), np.ndarray) else None,
            "prediction_shape": list(tensors["predictions"].shape) if isinstance(tensors.get("predictions"), np.ndarray) else None,
            "activation_layers": activation_meta,
            "saliency": {
                "available": saliency_array is not None,
                "kind": saliency_kind,
                "reason": saliency_reason,
                "shape": list(np.asarray(saliency_array).shape) if saliency_array is not None else None,
            },
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

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
            self._reset_activation_capture_window()
            return

        # PyTorch path: flush activation buffer populated by forward hooks
        activation_source = self.activation_buffer
        if not activation_source and self.cached_probe_activations:
            activation_source = {layer: [arr] for layer, arr in self.cached_probe_activations.items()}

        for layer_name, act_list in activation_source.items():
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
        self._reset_activation_capture_window()

    @staticmethod
    def detect_leakage(x, y):
        pass

    def log_batch_predictions(self, step, x, y, y_pred, loss):
        pred_dir = self.run_dir / "predictions"
        try:
            timestamp = time.time()
            x_np = self._extract_primary_input_array(x)
            y_np = self._to_numpy(y)
            if y_np is not None and getattr(y_np, "ndim", 0) > 0:
                y_np = y_np[: self.probe_examples]
            y_pred_np = self._to_numpy(y_pred)
            record = {"step": step, "timestamp": timestamp}
            if x_np is not None:
                record["input_shape"] = list(x_np.shape)
                record["input_modality_hint"] = self._infer_input_modality(x_np)
            if y_np is not None:
                record["target_shape"] = list(y_np.shape)
                if getattr(y_np, "ndim", 0) == 0:
                    record["y_true"] = y_np.tolist()
                else:
                    record["y_true"] = y_np[:256].tolist() if len(y_np) > 256 else y_np.tolist()
            if y_pred_np is not None:
                record["prediction_shape"] = list(y_pred_np.shape)
                if self._looks_like_class_scores(y_np, y_pred_np):
                    pred_classes = np.argmax(y_pred_np, axis=-1)
                    confidence = self._confidence_from_scores(y_pred_np)
                    record["y_pred"] = pred_classes[:256].tolist()
                    record["confidence"] = confidence[:256].tolist()
                    record["logits_sample"] = y_pred_np[:16].tolist()
                    record["prediction_type"] = "class_scores"
                else:
                    if getattr(y_pred_np, "ndim", 0) == 0:
                        record["y_pred"] = y_pred_np.tolist()
                    else:
                        record["y_pred"] = y_pred_np[:256].tolist()
                    record["prediction_type"] = "numeric_array" if getattr(y_pred_np, "ndim", 0) > 1 else "numeric_vector"
            if loss is not None:
                loss_np = self._to_numpy(loss)
                record["loss"] = float(loss) if np.isscalar(loss) else float(loss_np)
            filepath = pred_dir / f"probe_step_{step}.json"
            with open(filepath, "w") as f:
                json.dump(record, f, indent=2, default=str)

            if self.framework == "pytorch":
                probe_bundle = self._capture_pytorch_probe_bundle(x)
            elif self.framework == "tensorflow":
                probe_bundle = self._capture_tensorflow_probe_bundle(x)
            else:
                probe_bundle = {
                    "activation_arrays": {},
                    "prediction_array": None,
                    "confidence_array": None,
                    "logits_array": None,
                    "saliency_array": None,
                    "saliency_reason": "Probe capture is unavailable for this framework.",
                    "saliency_kind": None,
                }

            prediction_array = probe_bundle.get("prediction_array")
            confidence_array = probe_bundle.get("confidence_array")
            logits_array = probe_bundle.get("logits_array")
            activation_arrays = probe_bundle.get("activation_arrays") or {}
            self.cached_probe_activations = {
                layer: np.asarray(values)
                for layer, values in activation_arrays.items()
            }

            if prediction_array is None and y_pred_np is not None:
                y_pred_probe = y_pred_np[: self.probe_examples] if getattr(y_pred_np, "ndim", 0) > 0 else y_pred_np
                prediction_array, confidence_guess, logits_guess = self._reduce_prediction_array(y_pred_probe)
                if confidence_array is None:
                    confidence_array = confidence_guess
                if logits_array is None:
                    logits_array = logits_guess

            self._write_probe_bundle(
                step=step,
                timestamp=timestamp,
                primary_input=x_np,
                y_true_array=y_np,
                prediction_array=prediction_array,
                confidence_array=confidence_array,
                logits_array=logits_array,
                activation_arrays=activation_arrays,
                saliency_array=probe_bundle.get("saliency_array"),
                saliency_kind=probe_bundle.get("saliency_kind"),
                saliency_reason=probe_bundle.get("saliency_reason"),
            )
            self._reset_probe_window()
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


def _safe_positive_int(value):
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


try:
    import tensorflow as _tf_for_keras_cb

    _KerasCallbackBase = _tf_for_keras_cb.keras.callbacks.Callback
except (ImportError, AttributeError):
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
