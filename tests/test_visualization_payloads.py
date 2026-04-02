import json

import numpy as np

from gradglass.artifacts import ArtifactStore
from gradglass.visualizations import build_distributions_payload, build_embeddings_payload, build_saliency_payload


def _write_checkpoint(run_dir, step, tensors):
    checkpoint_dir = run_dir / "checkpoints"
    np.savez_compressed(str(checkpoint_dir / f"step_{step}.npz"), **tensors)
    with open(checkpoint_dir / f"step_{step}_meta.json", "w") as f:
        json.dump({"step": step, "timestamp": 0.0, "num_params": int(sum(arr.size for arr in tensors.values()))}, f)


def _write_probe_bundle(run_dir, step, arrays, meta):
    probe_dir = run_dir / "probes"
    np.savez_compressed(str(probe_dir / f"probe_step_{step}.npz"), **arrays)
    with open(probe_dir / f"probe_step_{step}.json", "w") as f:
        json.dump(meta, f, indent=2)


def test_visualization_payloads_and_saliency_fallback(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run_dir = store.ensure_run_dir("demo-run")

    _write_checkpoint(
        run_dir,
        step=5,
        tensors={
            "encoder.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            "encoder.bias": np.array([0.1, -0.2, 0.3], dtype=np.float32),
        },
    )

    probe_arrays = {
        "input": np.random.randn(4, 6).astype(np.float32),
        "targets": np.array([0, 1, 0, 1], dtype=np.int64),
        "predictions": np.array([0, 1, 1, 1], dtype=np.int64),
        "confidence": np.array([0.82, 0.91, 0.67, 0.74], dtype=np.float32),
        "saliency": np.abs(np.random.randn(4, 6)).astype(np.float32),
        "activation__encoder": np.random.randn(4, 6).astype(np.float32),
        "activation__conv1": np.random.randn(4, 3, 4, 4).astype(np.float32),
    }
    _write_probe_bundle(
        run_dir,
        step=7,
        arrays=probe_arrays,
        meta={
            "step": 7,
            "probe_examples": 4,
            "input_modality": "structured_data",
            "input_shape": [4, 6],
            "target_shape": [4],
            "prediction_shape": [4],
            "activation_layers": [
                {"layer": "encoder", "key": "activation__encoder", "shape": [4, 6]},
                {"layer": "conv1", "key": "activation__conv1", "shape": [4, 3, 4, 4]},
            ],
            "saliency": {"available": True, "kind": "structured", "reason": None, "shape": [4, 6]},
        },
    )

    distributions = build_distributions_payload(store, "demo-run")
    assert distributions["weights"]["available"] is True
    assert distributions["activations"]["available"] is True
    assert distributions["weights"]["step"] == 5
    assert {entry["layer"] for entry in distributions["activations"]["layers"]} == {"encoder", "conv1"}

    embeddings = build_embeddings_payload(store, "demo-run")
    assert embeddings["available"] is True
    assert embeddings["default_layer"] == "encoder"
    assert embeddings["layers"][0]["matrix_shape"][0] == 4
    assert len(embeddings["layers"][0]["projection"][0]) == 2
    assert any(layer["pooling"] == "spatial_mean" for layer in embeddings["layers"])

    saliency = build_saliency_payload(store, "demo-run")
    assert saliency["available"] is True
    assert saliency["modality"] == "structured"
    assert saliency["feature_importance"]

    legacy_run_dir = store.ensure_run_dir("legacy-run")
    with open(legacy_run_dir / "metadata.json", "w") as f:
        json.dump({"run_id": "legacy-run"}, f)
    fallback = build_saliency_payload(store, "legacy-run")
    assert fallback["available"] is False
    assert "saliency" in fallback["reason"].lower()
