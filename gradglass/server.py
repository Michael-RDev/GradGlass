from __future__ import annotations
import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gradglass.alerts import build_alert_snapshot
from gradglass.analysis.data_monitor import load_dataset_monitor_report
from gradglass.analysis.leakage import project_monitor_report_to_legacy_dict
from gradglass.artifacts import ArtifactStore
from gradglass.browser import schedule_url_open_detached
from gradglass.diff import full_diff, gradient_flow_analysis, prediction_diff, architecture_diff
from gradglass.evaluation import build_evaluation_payload
from gradglass.experiment_tracking import build_overview_snapshot, normalize_run_status
from gradglass.telemetry import collect_infrastructure_telemetry
from gradglass.visualizations import build_distributions_payload, build_embeddings_payload, build_saliency_payload


def get_overview_snapshot(store: ArtifactStore, run_id: str, metrics: Optional[list[dict[str, Any]]] = None):
    meta = store.get_run_metadata(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    metric_rows = metrics if metrics is not None else store.get_metrics(run_id)
    runtime_state = store.get_runtime_state(run_id)

    return build_overview_snapshot(
        run_id=run_id,
        metadata=meta,
        metrics=metric_rows,
        runtime_state=runtime_state,
    )


def create_app(store):
    app = FastAPI(title="GradGlass", description="Neural Network Transparency Engine", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    app.state.store = store

    @app.get("/api/runs")
    async def list_runs():
        runs = store.list_runs()
        for run in runs:
            run_id = run.get("run_id")
            if not run_id:
                run["status"] = normalize_run_status(run.get("status"))
                continue
            try:
                overview = build_overview_snapshot(
                    run_id=run_id,
                    metadata=run,
                    metrics=store.get_metrics(run_id),
                    runtime_state=store.get_runtime_state(run_id),
                )
                run["status"] = overview.get("status", normalize_run_status(run.get("status")))
                run["health_state"] = overview.get("health_state")
                run["heartbeat_ts"] = overview.get("heartbeat_ts")
            except Exception:
                run["status"] = normalize_run_status(run.get("status"))
        return {"runs": runs, "total": len(runs)}

    @app.get("/api/compare")
    async def compare_runs(run_ids: str = Query(..., description="Comma-separated list of run IDs")):
        ids = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
        result = {}
        for rid in ids:
            metrics = store.get_metrics(rid)
            result[rid] = {"metrics": metrics}
        return result

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id):
        meta = store.get_run_metadata(run_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        overview = build_overview_snapshot(
            run_id=run_id,
            metadata=meta,
            metrics=store.get_metrics(run_id),
            runtime_state=store.get_runtime_state(run_id),
        )
        meta["run_id"] = run_id
        meta["status"] = overview.get("status", normalize_run_status(meta.get("status")))
        meta["health_state"] = overview.get("health_state")
        meta["heartbeat_ts"] = overview.get("heartbeat_ts")
        meta["metrics_summary"] = store.get_latest_metrics(run_id)
        meta["num_checkpoints"] = len(store.list_checkpoints(run_id))
        return meta

    @app.get("/api/runs/{run_id}/metrics")
    async def get_metrics(run_id):
        metrics = store.get_metrics(run_id)
        return {"run_id": run_id, "metrics": metrics, "total": len(metrics)}

    @app.get("/api/runs/{run_id}/overview")
    async def get_overview(run_id):
        return get_overview_snapshot(store, run_id)

    @app.get("/api/runs/{run_id}/alerts")
    async def get_alerts(run_id):
        meta = store.get_run_metadata(run_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        metrics = store.get_metrics(run_id)
        runtime_state = store.get_runtime_state(run_id)
        overview = build_overview_snapshot(
            run_id=run_id,
            metadata=meta,
            metrics=metrics,
            runtime_state=runtime_state,
        )
        return build_alert_snapshot(
            store,
            run_id,
            metadata=meta,
            metrics=metrics,
            runtime_state=runtime_state,
            overview=overview,
        )

    @app.get("/api/runs/{run_id}/checkpoints")
    async def list_checkpoints(run_id):
        checkpoints = store.list_checkpoints(run_id)
        return {"run_id": run_id, "checkpoints": checkpoints}

    @app.get("/api/runs/{run_id}/diff")
    async def compute_diff(
        run_id,
        a=Query(..., description="Step number for checkpoint A"),
        b=Query(..., description="Step number for checkpoint B"),
        include_deltas=Query(False, description="Include delta histograms"),
    ):
        try:
            weights_a = store.load_checkpoint(run_id, a)
            weights_b = store.load_checkpoint(run_id, b)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        result = full_diff(weights_a=weights_a, weights_b=weights_b, run_id=run_id, step_a=a, step_b=b)
        return result.to_dict(include_deltas=include_deltas)

    @app.get("/api/runs/{run_id}/gradients")
    async def get_gradients(run_id):
        summaries = store.get_gradient_summaries(run_id)
        analysis = gradient_flow_analysis(summaries)
        return {"run_id": run_id, "summaries": summaries, "analysis": analysis}

    @app.get("/api/runs/{run_id}/activations")
    async def get_activations(run_id):
        stats = store.get_activation_stats(run_id)
        return {"run_id": run_id, "activations": stats}

    @app.get("/api/runs/{run_id}/distributions")
    async def get_distributions(run_id, step: Optional[int] = Query(None, description="Probe step to inspect")):
        if not store.get_run_dir(run_id).exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return build_distributions_payload(store, run_id, step=step)

    @app.get("/api/runs/{run_id}/saliency")
    async def get_saliency(run_id, step: Optional[int] = Query(None, description="Probe step to inspect")):
        if not store.get_run_dir(run_id).exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return build_saliency_payload(store, run_id, step=step)

    @app.get("/api/runs/{run_id}/embeddings")
    async def get_embeddings(run_id, step: Optional[int] = Query(None, description="Probe step to inspect")):
        if not store.get_run_dir(run_id).exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return build_embeddings_payload(store, run_id, step=step)

    @app.get("/api/runs/{run_id}/predictions")
    async def get_predictions(run_id):
        predictions = store.get_predictions(run_id)
        diffs = []
        for i in range(1, len(predictions)):
            diff = prediction_diff(predictions[i - 1], predictions[i])
            diffs.append(diff)
        return {"run_id": run_id, "predictions": predictions, "diffs": diffs}

    @app.get("/api/runs/{run_id}/architecture")
    async def get_architecture(run_id):
        arch = store.get_architecture(run_id)
        if arch is None:
            raise HTTPException(status_code=404, detail="No architecture data found")
        return arch

    @app.get("/api/runs/{run_id}/analysis")
    async def get_analysis(run_id):
        run_dir = store.get_run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        report_path = run_dir / "analysis" / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        from gradglass.analysis.report import PostRunReport

        try:
            report = PostRunReport.generate(run_id=run_id, store=store, run_dir=run_dir, save=True, print_summary=False)
            return report.to_dict()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    @app.get("/api/runs/{run_id}/freeze_code")
    async def get_freeze_code(run_id):
        run_dir = store.get_run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        summaries = store.get_gradient_summaries(run_id)
        arch = store.get_architecture(run_id)

        layer_norms: dict[str, list[float]] = {}
        for entry in summaries:
            for layer, data in entry.get("layers", {}).items():
                norm = data.get("norm")
                if norm is not None and not (norm != norm):  # isnan check
                    layer_norms.setdefault(layer, []).append(norm)

        if not layer_norms:
            return {
                "run_id": run_id,
                "candidates": [],
                "pytorch_code": "# No gradient data found — run with gradients='summary' in watch()",
                "tensorflow_code": "# No gradient data found",
                "message": "No gradient summaries found for this run.",
            }

        mean_norms = {layer: float(sum(norms) / len(norms)) for layer, norms in layer_norms.items()}
        max_mean = max(mean_norms.values()) if mean_norms else 1.0
        threshold = max_mean * 0.01

        candidates = sorted(
            [
                {
                    "layer": layer,
                    "mean_grad_norm": round(norm, 8),
                    "relative_norm": round(norm / max(max_mean, 1e-12), 6),
                }
                for layer, norm in mean_norms.items()
                if norm < threshold
            ],
            key=lambda x: x["mean_grad_norm"],
        )

        pytorch_lines = [
            "import torch",
            "",
            "# ❄️ GradGlass: Freeze layers with low gradient activity",
            "# These layers received <1% of the maximum gradient norm — good freeze candidates.",
            "#",
            "# Option A: Freeze specific layers",
        ]
        for c in candidates[:8]:
            layer_path = c["layer"].replace(".", ".")
            pytorch_lines.append(f"# for param in model.{layer_path}.parameters():")
            pytorch_lines.append(f"#     param.requires_grad_(False)")

        pytorch_lines += [
            "",
            "# Option B: Remove frozen layers from optimizer",
            "# optimizer = torch.optim.Adam(",
            "#     filter(lambda p: p.requires_grad, model.parameters()),",
            "#     lr=your_lr",
            "# )",
            "",
            "# Option C: Do nothing (keep all layers trainable — current state)",
        ]

        tensorflow_lines = ["# ❄️ GradGlass: Freeze layers with low gradient activity (Keras/TF)"]
        for c in candidates[:8]:
            tensorflow_lines.append(f"# model.get_layer('{c['layer']}').trainable = False")
        tensorflow_lines += ["", "# Or do nothing to keep all layers trainable"]

        return {
            "run_id": run_id,
            "candidates": candidates[:10],
            "total_candidates": len(candidates),
            "pytorch_code": "\n".join(pytorch_lines),
            "tensorflow_code": "\n".join(tensorflow_lines),
            "message": f"Found {len(candidates)} freeze candidate(s) out of {len(mean_norms)} layers."
            if candidates
            else "All layers are receiving meaningful gradients — no freeze candidates.",
        }

    @app.get("/api/runs/{run_id}/architecture/diff")
    async def get_architecture_diff(run_id, compare_run_id=Query(None, description="Run to compare against")):
        arch_a = store.get_architecture(run_id)
        if arch_a is None:
            raise HTTPException(status_code=404, detail="No architecture for source run")
        if compare_run_id:
            arch_b = store.get_architecture(compare_run_id)
            if arch_b is None:
                raise HTTPException(status_code=404, detail="No architecture for comparison run")
        else:
            arch_b = arch_a
        diff = architecture_diff(arch_a, arch_b)
        return diff

    @app.get("/api/runs/{run_id}/leakage")
    async def get_leakage_report(run_id):
        run_dir = store.get_run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        report_path = run_dir / "analysis" / "leakage_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        monitor_report = load_dataset_monitor_report(run_dir)
        if monitor_report is not None:
            return project_monitor_report_to_legacy_dict(monitor_report)
        raise HTTPException(
            status_code=404,
            detail="No leakage report found for this run. Use run.check_leakage(), run.check_leakage_from_loaders(), or run.monitor_dataset(...).finalize() to generate one.",
        )

    @app.get("/api/runs/{run_id}/data-monitor")
    async def get_data_monitor(run_id):
        run_dir = store.get_run_dir(run_id)
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        report = load_dataset_monitor_report(run_dir)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="No dataset monitor report found for this run. Use run.monitor_dataset(...).finalize() or run.check_leakage(...) to generate one.",
            )
        return report.model_dump(mode="json")

    class MutationRequest(BaseModel):
        operation: str
        target_layer: str
        params: dict[str, Any] = {}

    @app.post("/api/runs/{run_id}/architecture/mutate")
    async def mutate_architecture(run_id: str, mutation: MutationRequest):
        arch = store.get_architecture(run_id)
        if arch is None:
            raise HTTPException(status_code=404, detail="No architecture data found")
        import copy

        draft = copy.deepcopy(arch)
        result = apply_mutation(draft, mutation)
        if result.get("valid"):
            arch_path = store.get_run_dir(run_id) / "model_structure.json"
            with open(arch_path, "w") as f:
                json.dump(result["draft"], f, indent=2)
        return result

    @app.get("/api/runs/{run_id}/distributed")
    async def get_distributed(run_id):
        dist_info = store.get_distributed_info(run_id)
        ranks = store.list_ranks(run_id)
        return {"run_id": run_id, "distributed_index": dist_info, "ranks": ranks}

    @app.get("/api/runs/{run_id}/infrastructure")
    async def get_infrastructure(run_id):
        meta = store.get_run_metadata(run_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return collect_infrastructure_telemetry(store, run_id)

    @app.get("/api/runs/{run_id}/eval")
    async def get_eval_metrics(run_id):
        meta = store.get_run_metadata(run_id)
        if meta is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

        predictions = store.get_predictions(run_id)
        metrics = store.get_metrics(run_id)
        if not predictions and not metrics:
            raise HTTPException(
                status_code=404,
                detail="No evaluation data found. Log predictions with run.log_batch() or metrics with run.log().",
            )

        report = build_evaluation_payload(run_id, metadata=meta, metrics=metrics, predictions=predictions)
        return {"run_id": run_id, "evaluations": report["evaluations"], "report": report}

    @app.websocket("/api/runs/{run_id}/stream")
    async def stream_metrics(websocket: WebSocket, run_id: str):
        await websocket.accept()
        existing_metrics = store.get_metrics(run_id)
        last_step = max((m.get("step", 0) for m in existing_metrics), default=0)
        try:
            while True:
                metrics = store.get_metrics(run_id)
                new_metrics = [m for m in metrics if m.get("step", 0) > last_step]
                if new_metrics:
                    last_step = new_metrics[-1].get("step", last_step)
                    await websocket.send_json({"type": "metrics_update", "data": new_metrics})
                try:
                    overview = get_overview_snapshot(store, run_id, metrics=metrics)
                    await websocket.send_json({"type": "overview_update", "data": overview})
                    meta = store.get_run_metadata(run_id)
                    if meta is not None:
                        alerts = build_alert_snapshot(
                            store,
                            run_id,
                            metadata=meta,
                            metrics=metrics,
                            runtime_state=store.get_runtime_state(run_id),
                            overview=overview,
                        )
                        await websocket.send_json({"type": "alerts_update", "data": alerts})
                except HTTPException:
                    await websocket.send_json(
                        {
                            "type": "overview_update",
                            "data": {
                                "run_id": run_id,
                                "status": "failed",
                                "health_state": "FAILED",
                                "total_steps_source": "unknown",
                                "eta_s": None,
                                "eta_reason": "Run not found",
                            },
                        }
                    )
                    await websocket.send_json(
                        {
                            "type": "alerts_update",
                            "data": {
                                "run_id": run_id,
                                "status": "failed",
                                "health_state": "FAILED",
                                "summary": {
                                    "total": 0,
                                    "critical": 0,
                                    "high": 0,
                                    "medium": 0,
                                    "low": 0,
                                    "high_severity": 0,
                                    "warnings": 0,
                                    "fail_count": 0,
                                    "warn_count": 0,
                                    "health_state": "FAILED",
                                    "health_reason": "Run not found",
                                    "updated_at": time.time(),
                                    "top_alert_id": None,
                                },
                                "alerts": [],
                            },
                        }
                    )
                    break
                if str((overview or {}).get("status") or "").strip().lower() in {"complete", "completed", "finished", "failed", "cancelled", "interrupted"}:
                    break
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    dashboard_dir = Path(__file__).parent / "dashboard" / "dist"
    if dashboard_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(dashboard_dir / "assets")), name="assets")

        @app.get("/{path:path}")
        async def serve_dashboard(path):
            file_path = dashboard_dir / path
            if file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(dashboard_dir / "index.html"))
    else:

        @app.get("/")
        async def dashboard_placeholder():
            return HTMLResponse(
                '\n            <!DOCTYPE html>\n            <html>\n            <head><title>GradGlass</title></head>\n            <body style="font-family: system-ui; display:flex; justify-content:center; align-items:center; height:100vh; margin:0; background:#0f172a; color:#e2e8f0;">\n                <div style="text-align:center">\n                    <h1>🔬 GradGlass v1.0</h1>\n                    <p>Dashboard build not found. API is running.</p>\n                    <p>Try: <code>GET /api/runs</code></p>\n                </div>\n            </body>\n            </html>\n            '
            )

    return app


def apply_mutation(draft, mutation):
    layers = {l["id"]: l for l in draft.get("layers", [])}
    edges = draft.get("edges", [])
    if mutation.operation == "freeze":
        if mutation.target_layer not in layers:
            return {"valid": False, "error": f"Layer '{mutation.target_layer}' not found"}
        layers[mutation.target_layer]["trainable"] = False
        return {"valid": True, "draft": draft, "message": f"Layer '{mutation.target_layer}' frozen"}
    elif mutation.operation == "unfreeze":
        if mutation.target_layer not in layers:
            return {"valid": False, "error": f"Layer '{mutation.target_layer}' not found"}
        layers[mutation.target_layer]["trainable"] = True
        return {"valid": True, "draft": draft, "message": f"Layer '{mutation.target_layer}' unfrozen"}
    elif mutation.operation == "remove":
        if mutation.target_layer not in layers:
            return {"valid": False, "error": f"Layer '{mutation.target_layer}' not found"}
        predecessors = [e[0] for e in edges if e[1] == mutation.target_layer]
        successors = [e[1] for e in edges if e[0] == mutation.target_layer]
        new_edges = [e for e in edges if mutation.target_layer not in e]
        for pred in predecessors:
            for succ in successors:
                new_edges.append([pred, succ])
        draft["edges"] = new_edges
        draft["layers"] = [l for l in draft["layers"] if l["id"] != mutation.target_layer]
        return {"valid": True, "draft": draft, "message": f"Layer '{mutation.target_layer}' removed"}
    elif mutation.operation == "add":
        new_layer = {
            "id": mutation.target_layer,
            "type": mutation.params.get("type", "Linear"),
            "params": mutation.params,
            "param_count": 0,
            "trainable": True,
        }
        draft["layers"].append(new_layer)
        if "after" in mutation.params:
            after_id = mutation.params["after"]
            new_edges = []
            for e in edges:
                new_edges.append(e)
                if e[0] == after_id:
                    new_edges.append([after_id, mutation.target_layer])
                    new_edges.append([mutation.target_layer, e[1]])
            draft["edges"] = new_edges
        return {"valid": True, "draft": draft, "message": f"Layer '{mutation.target_layer}' added"}
    return {"valid": False, "error": f"Unknown operation: {mutation.operation}"}


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 10.0, interval: float = 0.05):
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=interval):
                return True
        except OSError:
            _time.sleep(interval)
    return False


def start_server(app, port=0):
    if port == 0:
        port = find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server("127.0.0.1", port)
    return port


def start_server_blocking(app, port=8432, open_browser=True):
    if open_browser:
        schedule_url_open_detached(f"http://localhost:{port}", delay_s=1.5, force_reload=True)
    print(f"🔬 GradGlass server running at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the GradGlass dashboard server.")
    parser.add_argument("--root", default=None, help="Artifact store root. Defaults to ./gg_artifacts in the cwd.")
    parser.add_argument("--port", type=int, default=8432, help="Port to bind the GradGlass dashboard server.")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the dashboard URL in a browser after the server starts.",
    )
    args = parser.parse_args()

    store = ArtifactStore(root=args.root)
    app = create_app(store)
    start_server_blocking(app, port=args.port, open_browser=args.open_browser)


if __name__ == "__main__":
    main()
