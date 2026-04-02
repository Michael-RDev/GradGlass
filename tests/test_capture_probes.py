import json

import numpy as np

from gradglass.capture import CaptureEngine


class _FakeModel:
    training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def zero_grad(self, *args, **kwargs):
        return None


def test_probe_bundle_capture_and_activation_window_reset(tmp_path):
    run_dir = tmp_path / "run"
    engine = CaptureEngine(
        model=_FakeModel(),
        optimizer=None,
        framework="pytorch",
        run_dir=run_dir,
        config={
            "layers": "trainable",
            "activations": "auto",
            "gradients": "summary",
            "saliency": "off",
            "probe_examples": 4,
            "sample_batches": 1,
        },
    )

    engine.activation_buffer = {"encoder": [np.ones((2, 4), dtype=np.float32)]}
    engine.activation_capture_counts = {"encoder": 1}
    engine._capture_pytorch_probe_bundle = lambda x: {  # noqa: SLF001
        "activation_arrays": {"encoder": np.arange(24, dtype=np.float32).reshape(4, 6)},
        "prediction_array": np.array([0, 1, 0, 1], dtype=np.int64),
        "confidence_array": np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
        "logits_array": np.random.randn(4, 3).astype(np.float32),
        "saliency_array": None,
        "saliency_reason": "Saliency capture was disabled for this run.",
        "saliency_kind": None,
    }

    engine.log_batch_predictions(
        step=1,
        x=np.random.randn(8, 4).astype(np.float32),
        y=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        y_pred=np.random.randn(8, 3).astype(np.float32),
        loss=1.0,
    )

    probe_meta_path = run_dir / "probes" / "probe_step_1.json"
    probe_data_path = run_dir / "probes" / "probe_step_1.npz"
    assert probe_meta_path.exists()
    assert probe_data_path.exists()
    assert engine.activation_capture_counts == {}
    assert engine.activation_buffer == {}

    with open(probe_meta_path) as f:
        meta = json.load(f)
    assert meta["probe_examples"] == 4
    assert meta["saliency"]["available"] is False
    assert meta["activation_layers"]

    with np.load(str(probe_data_path), allow_pickle=False) as data:
        assert "input" in data.files
        assert "predictions" in data.files
        assert "activation__encoder" in data.files

    engine.cleanup()
