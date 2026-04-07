import asyncio

import numpy as np
import pytest

from gradglass.artifacts import ArtifactStore
from gradglass.run import Run
from gradglass.server import create_app


class _FakeExplanation:
    def __init__(self, values):
        self.values = values


def test_run_log_shap_persists_ranked_summary_from_numpy(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run = Run(name="shap-numpy", store=store, auto_open=False)

    payload = run.log_shap(
        ["age", "income", "score"], np.array([[0.1, -0.6, 0.2], [0.5, -0.4, 0.1]], dtype=np.float32), top_k=2
    )

    assert payload["run_id"] == run.run_id
    assert payload["available"] is True
    assert payload["feature_count"] == 3
    assert payload["top_k"] == 2
    assert [item["feature"] for item in payload["summary_plot"]] == ["income", "age"]
    assert payload["summary_plot"][0]["mean_shap"] == 0.5

    saved = store.get_shap(run.run_id)
    assert saved == payload


def test_run_log_shap_accepts_explanation_like_objects(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run = Run(name="shap-explanation", store=store, auto_open=False)

    explanation = _FakeExplanation(
        np.array([[[0.2, 0.1, -0.3], [0.6, -0.1, 0.2]], [[0.1, -0.2, 0.4], [0.5, 0.1, -0.1]]], dtype=np.float32)
    )

    payload = run.log_shap(["signal_left", "signal_right"], explanation, message="Custom SHAP aggregation.", top_k=1)

    assert payload["message"] == "Custom SHAP aggregation."
    assert payload["summary_plot"] == [{"feature": "signal_right", "mean_shap": 0.266667}]


def test_run_log_lime_persists_normalized_samples(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run = Run(name="lime-samples", store=store, auto_open=False)

    payload = run.log_lime(
        [
            {
                "prediction": "positive",
                "probability": np.float32(0.91),
                "explanation": [
                    {"feature": "income", "weight": np.float64(0.42)},
                    {"feature": "debt_ratio", "weight": np.float64(-0.17)},
                ],
            }
        ],
        message="Local explanation samples.",
    )

    assert payload["run_id"] == run.run_id
    assert payload["available"] is True
    assert payload["message"] == "Local explanation samples."
    assert payload["sample_count"] == 1
    assert payload["samples"][0]["index"] == 0
    assert payload["samples"][0]["prediction"] == "positive"
    assert payload["samples"][0]["probability"] == pytest.approx(0.91)
    assert payload["samples"][0]["explanation"][1] == {"feature": "debt_ratio", "weight": -0.17}

    saved = store.get_lime(run.run_id)
    assert saved == payload


def test_shap_and_lime_endpoints_return_saved_artifacts(tmp_path):
    store = ArtifactStore(root=tmp_path)
    store.save_shap(
        "api-run", {"message": "Saved SHAP payload.", "summary_plot": [{"feature": "income", "mean_shap": 0.8}]}
    )
    store.save_lime(
        "api-run",
        {
            "message": "Saved LIME payload.",
            "samples": [
                {
                    "index": 0,
                    "prediction": "positive",
                    "probability": 0.88,
                    "explanation": [{"feature": "income", "weight": 0.33}],
                }
            ],
        },
    )

    app = create_app(store)
    route_map = {
        route.path: route.endpoint
        for route in app.router.routes
        if hasattr(route, "path") and hasattr(route, "endpoint")
    }

    shap_payload = asyncio.run(route_map["/api/runs/{run_id}/shap"]("api-run"))
    lime_payload = asyncio.run(route_map["/api/runs/{run_id}/lime"]("api-run"))

    assert shap_payload["available"] is True
    assert shap_payload["summary_plot"][0]["feature"] == "income"
    assert lime_payload["available"] is True
    assert lime_payload["samples"][0]["explanation"][0]["weight"] == 0.33


def test_architecture_mutation_route_accepts_json_body(tmp_path):
    app = create_app(ArtifactStore(root=tmp_path))
    route = next(
        route for route in app.router.routes if getattr(route, "path", None) == "/api/runs/{run_id}/architecture/mutate"
    )

    assert [param.name for param in route.dependant.body_params] == ["mutation"]
    assert route.body_field is not None
    assert route.body_field.name == "mutation"
