from __future__ import annotations

import json

from gradglass.analysis.builtins import (
    test_grad_exploding as run_grad_exploding_check,
    test_grad_nan_inf as run_grad_nan_inf_check,
    test_grad_vanishing as run_grad_vanishing_check,
)
from gradglass.analysis.registry import TestContext as AnalysisContext, TestStatus as AnalysisStatus
from gradglass.analysis.report import PostRunReport
from gradglass.artifacts import ArtifactStore


def _gradient_summary(step: int, layers: dict[str, dict[str, float]]) -> dict:
    return {"step": step, "layers": layers}


def _context(tmp_path, gradient_summaries):
    return AnalysisContext(run_id="gradient-suite", run_dir=tmp_path, store=None, gradient_summaries=gradient_summaries)


def _write_gradient_summary(run_dir, step: int, layers: dict[str, dict[str, float]]) -> None:
    grad_dir = run_dir / "gradients"
    grad_dir.mkdir(parents=True, exist_ok=True)
    with open(grad_dir / f"summaries_step_{step}.json", "w") as handle:
        json.dump(layers, handle, indent=2)


def test_healthy_gradient_summaries_are_treated_as_healthy(tmp_path):
    summaries = [
        _gradient_summary(1, {"encoder.weight": {"mean": 0.02, "var": 0.3, "max": 0.8, "norm": 0.5}}),
        _gradient_summary(2, {"encoder.weight": {"mean": 0.03, "var": 0.2, "max": 0.7, "norm": 0.6}}),
    ]
    ctx = _context(tmp_path, summaries)

    assert run_grad_nan_inf_check(ctx).status == AnalysisStatus.PASS
    assert run_grad_vanishing_check(ctx).status == AnalysisStatus.PASS
    assert run_grad_exploding_check(ctx).status == AnalysisStatus.PASS


def test_nan_or_inf_gradients_are_not_healthy(tmp_path):
    summaries = [_gradient_summary(1, {"encoder.weight": {"mean": float("nan"), "var": 0.2, "max": 0.4, "norm": 0.5}})]
    result = run_grad_nan_inf_check(_context(tmp_path, summaries))

    assert result.status == AnalysisStatus.FAIL
    assert result.details["total"] >= 1


def test_near_zero_gradient_norms_are_not_healthy(tmp_path):
    summaries = [_gradient_summary(1, {"encoder.weight": {"mean": 0.0, "var": 0.2, "max": 0.1, "norm": 0.0}})]
    result = run_grad_vanishing_check(_context(tmp_path, summaries))

    assert result.status == AnalysisStatus.WARN
    assert result.details["total"] >= 1


def test_extreme_gradient_norms_are_not_healthy(tmp_path):
    summaries = [_gradient_summary(1, {"encoder.weight": {"mean": 0.5, "var": 0.8, "max": 50.0, "norm": 150.0}})]
    result = run_grad_exploding_check(_context(tmp_path, summaries))

    assert result.status == AnalysisStatus.WARN
    assert result.details["total"] >= 1


def test_report_generation_surfaces_flagged_unhealthy_gradient_layers(tmp_path):
    store = ArtifactStore(root=tmp_path)
    run_dir = store.ensure_run_dir("gradient-report")

    _write_gradient_summary(
        run_dir,
        step=1,
        layers={
            "dead.weight": {"mean": 0.0, "var": 0.2, "max": 0.0, "norm": 0.0},
            "exploding.weight": {"mean": 101.0, "var": 0.5, "max": 50.0, "norm": 150.0},
            "healthy.weight": {"mean": 0.04, "var": 0.2, "max": 0.8, "norm": 0.7},
        },
    )

    report = PostRunReport.generate(
        run_id="gradient-report",
        store=store,
        run_dir=run_dir,
        tests=["GRAD_NAN_INF", "GRAD_VANISHING", "GRAD_EXPLODING"],
        save=True,
        print_summary=False,
    )

    flagged_layers = {entry["layer"] for entry in report.gradient_flow_analysis["flagged_layers"]}

    assert report.tests["failed"] == 0
    assert report.tests["warned"] == 2
    assert report.gradient_flow_analysis["flagged"] == 2
    assert flagged_layers == {"dead.weight", "exploding.weight"}
    assert (run_dir / "analysis" / "report.json").exists()
    assert (run_dir / "analysis" / "tests.jsonl").exists()
