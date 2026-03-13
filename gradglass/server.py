from __future__ import annotations
import asyncio
import json
import socket
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from gradglass.artifacts import ArtifactStore
from gradglass.diff import full_diff, gradient_flow_analysis, prediction_diff, architecture_diff
from gradglass.experiment_tracking import build_overview_snapshot


def get_overview_snapshot(store: ArtifactStore, run_id: str, metrics: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    meta = store.get_run_metadata(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    metric_rows = metrics if metrics is not None else store.get_metrics(run_id)
    runtime_state = store.get_runtime_state(run_id)
    sklearn_diagnostics = store.get_sklearn_diagnostics(run_id)

    return build_overview_snapshot(
        run_id=run_id,
        metadata=meta,
        metrics=metric_rows,
        runtime_state=runtime_state,
        sklearn_diagnostics=sklearn_diagnostics,
    )


def create_app(store):
    app = FastAPI(title="GradGlass", description="Neural Network Transparency Engine — Local API", version="1.0.0")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
    )
    app.state.store = store

    @app.get("/api/runs")
    async def list_runs():
        runs = store.list_runs()
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
        meta["run_id"] = run_id
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
        metrics = store.get_metrics(run_id)
        alerts = []
        if metrics:
            for m in metrics:
                if "loss" in m:
                    val = m["loss"]
                    inf_nan = False
                    if val is None:
                        inf_nan = True
                    elif isinstance(val, float) and (val != val or val == float("inf") or val == float("-inf")):
                        inf_nan = True
                    if inf_nan:
                        alerts.append({"severity": "high", "title": "NaN or Inf Loss", "message": f"Loss became NaN/Inf at step {m.get('step')}"})
                        break
                        
            if len(metrics) > 5:
                # Simple overfitting check
                recent = metrics[-5:]
                train_losses = [m["loss"] for m in recent if "loss" in m and m["loss"] is not None]
                val_losses = [m.get("val_loss") for m in recent if m.get("val_loss") is not None]
                if len(train_losses) == 5 and len(val_losses) == 5:
                    if train_losses[0] > train_losses[-1] and val_losses[0] < val_losses[-1]:
                        alerts.append({"severity": "medium", "title": "Possible Overfitting", "message": "Validation loss is increasing while training loss is decreasing."})
                
        summaries = store.get_gradient_summaries(run_id)
        if summaries:
            last_summary = summaries[-1]
            step = last_summary.get("step")
            for layer, data in last_summary.get("layers", {}).items():
                if data.get("norm", 0) > 10.0:
                    alerts.append({"severity": "warning", "title": "Large Gradients", "message": f"Potentially exploding gradient (norm {data.get('norm'):.2f}) at step {step} in {layer}"})
                    break
        
        return {"run_id": run_id, "alerts": alerts}

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
            [{"layer": layer, "mean_grad_norm": round(norm, 8), "relative_norm": round(norm / max(max_mean, 1e-12), 6)}
             for layer, norm in mean_norms.items() if norm < threshold],
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

        tensorflow_lines = [
            "# ❄️ GradGlass: Freeze layers with low gradient activity (Keras/TF)",
        ]
        for c in candidates[:8]:
            tensorflow_lines.append(f"# model.get_layer('{c['layer']}').trainable = False")
        tensorflow_lines += ["", "# Or do nothing to keep all layers trainable"]

        return {
            "run_id": run_id,
            "candidates": candidates[:10],
            "total_candidates": len(candidates),
            "pytorch_code": "\n".join(pytorch_lines),
            "tensorflow_code": "\n".join(tensorflow_lines),
            "message": f"Found {len(candidates)} freeze candidate(s) out of {len(mean_norms)} layers." if candidates else "All layers are receiving meaningful gradients — no freeze candidates.",
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
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No leakage report found for this run. Use run.check_leakage() or run.check_leakage_from_loaders() to generate one.",
            )
        with open(report_path) as f:
            return json.load(f)

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

    @app.get("/api/runs/{run_id}/eval")
    async def get_eval_metrics(run_id):
        predictions = store.get_predictions(run_id)
        if not predictions:
            raise HTTPException(
                status_code=404, detail="No evaluation data found. Log predictions using run.log_batch()."
            )

        results = []
        for pred in predictions:
            step = pred.get("step")
            y_true = np.array(pred.get("y_true", []))
            y_pred = np.array(pred.get("y_pred", []))

            if len(y_true) == 0 or len(y_pred) == 0:
                continue

            y_true_float = y_true.astype(float)
            y_pred_float = y_pred.astype(float)
            is_classification = np.all(np.mod(y_true_float, 1) == 0) and np.all(np.mod(y_pred_float, 1) == 0)

            step_eval = {"step": step, "is_classification": bool(is_classification)}

            if is_classification:
                # Basic classification metrics
                correct = np.sum(y_true == y_pred)
                total = len(y_true)
                step_eval["accuracy"] = float(correct / total) if total > 0 else 0.0
                step_eval["support"] = total

                # Confusion matrix
                classes = np.unique(np.concatenate([y_true, y_pred]))
                num_classes = len(classes)
                class_to_idx = {c: i for i, c in enumerate(classes)}
                cm = np.zeros((num_classes, num_classes), dtype=int)
                for t, p in zip(y_true, y_pred):
                    cm[class_to_idx[t], class_to_idx[p]] += 1

                step_eval["confusion_matrix"] = {"classes": [int(c) for c in classes], "matrix": cm.tolist()}

                # Per class precision/recall/f1
                per_class = []
                macro_f1, macro_p, macro_r = 0, 0, 0
                for i, cls in enumerate(classes):
                    tp = cm[i, i]
                    fp = np.sum(cm[:, i]) - tp
                    fn = np.sum(cm[i, :]) - tp

                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                    macro_p += p
                    macro_r += r
                    macro_f1 += f1

                    per_class.append(
                        {
                            "class": int(cls),
                            "precision": float(p),
                            "recall": float(r),
                            "f1": float(f1),
                            "support": int(np.sum(cm[i, :])),
                        }
                    )

                step_eval["per_class"] = per_class
                if num_classes > 0:
                    step_eval["macro_f1"] = float(macro_f1 / num_classes)
                    step_eval["macro_precision"] = float(macro_p / num_classes)
                    step_eval["macro_recall"] = float(macro_r / num_classes)
            else:
                error = y_true_float - y_pred_float
                mse = np.mean(error**2)
                mae = np.mean(np.abs(error))
                step_eval["mse"] = float(mse)
                step_eval["rmse"] = float(np.sqrt(mse))
                step_eval["mae"] = float(mae)

            results.append(step_eval)

        return {"run_id": run_id, "evaluations": results}

    @app.websocket("/api/runs/{run_id}/stream")
    async def stream_metrics(websocket, run_id):
        await websocket.accept()
        last_step = 0
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


def _wait_for_server(host: str, port: int, timeout: float = 10.0, interval: float = 0.05) -> bool:
    """Poll until the server's TCP port accepts connections or timeout expires."""
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

        def open_fn():
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_fn, daemon=True).start()
    print(f"🔬 GradGlass server running at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
