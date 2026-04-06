from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
import urllib.parse

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gradglass.artifacts import resolve_default_root
from gradglass.run import Run

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from _showcase_support import (
    build_tabular_run,
    configure_store,
    create_distributed_artifacts,
    degrade_run_for_showcase,
    resolve_server_port,
)


SHOWCASE_RUN_NAMES = {
    "primary": "dashboard_showcase_primary",
    "baseline": "dashboard_showcase_baseline",
    "problem": "dashboard_showcase_problem",
}


def _latest_runs_by_name(store, names: dict[str, str]) -> dict[str, Run]:
    indexed: dict[str, dict[str, Any]] = {}
    for meta in store.list_runs():
        name = meta.get("name")
        if name not in names.values():
            continue
        previous = indexed.get(name)
        if previous is None or float(meta.get("start_time_epoch") or 0.0) >= float(previous.get("start_time_epoch") or 0.0):
            indexed[name] = meta

    resolved = {}
    for key, name in names.items():
        meta = indexed.get(name)
        if meta is not None:
            resolved[key] = store.get_run_metadata(meta["run_id"]) or meta
            resolved[key]["run_id"] = meta["run_id"]
    return {key: Run.from_existing(value["run_id"], store=store) for key, value in resolved.items()}


def create_showcase_runs(root: Path, *, keep_existing: bool = False) -> dict[str, Any]:
    store = configure_store(root)
    if keep_existing:
        existing = _latest_runs_by_name(store, SHOWCASE_RUN_NAMES)
        if len(existing) == len(SHOWCASE_RUN_NAMES):
            return {"store": store, "runs": existing}

    baseline = build_tabular_run(
        store,
        name=SHOWCASE_RUN_NAMES["baseline"],
        width=18,
        seed=11,
        include_extra_artifacts=True,
        finish_run=True,
        epochs=4,
    )
    primary = build_tabular_run(
        store,
        name=SHOWCASE_RUN_NAMES["primary"],
        width=28,
        seed=17,
        include_extra_artifacts=True,
        finish_run=True,
        epochs=6,
    )
    problem = build_tabular_run(
        store,
        name=SHOWCASE_RUN_NAMES["problem"],
        width=12,
        seed=29,
        include_extra_artifacts=True,
        finish_run=False,
        epochs=4,
    )

    create_distributed_artifacts(store, primary["run"].run_id, healthy_nodes=2)
    create_distributed_artifacts(store, problem["run"].run_id, healthy_nodes=1)
    degrade_run_for_showcase(
        problem["run"],
        val_x=problem["val_x"],
        val_y=problem["val_y"],
        failure_message="Synthetic showcase failure: validation drift and runtime instability detected.",
    )

    return {
        "store": store,
        "runs": {
            "baseline": baseline["run"],
            "primary": primary["run"],
            "problem": problem["run"],
        },
    }


def build_guided_tour(base_url: str, runs: dict[str, Run]) -> list[tuple[str, str]]:
    primary_id = urllib.parse.quote(runs["primary"].run_id, safe="")
    problem_id = urllib.parse.quote(runs["problem"].run_id, safe="")
    return [
        ("Home", base_url),
        ("Primary overview", f"{base_url}/run/{primary_id}/overview"),
        ("Primary training", f"{base_url}/run/{primary_id}/training"),
        ("Primary evaluation", f"{base_url}/run/{primary_id}/evaluation"),
        ("Primary internals", f"{base_url}/run/{primary_id}/internals"),
        ("Primary data", f"{base_url}/run/{primary_id}/data"),
        ("Primary interpretability", f"{base_url}/run/{primary_id}/interpretability"),
        ("Primary compare", f"{base_url}/run/{primary_id}/compare"),
        ("Problem alerts", f"{base_url}/run/{problem_id}/alerts"),
        ("Problem infrastructure", f"{base_url}/run/{problem_id}/infrastructure"),
    ]


def print_guided_tour(base_url: str, runs: dict[str, Run]) -> None:
    print("\nDashboard Showcase")
    print("=" * 72)
    print("Runs:")
    for label, run in runs.items():
        print(f"  - {label:<8} {run.run_id}")
    print("\nGuided tour:")
    for label, url in build_guided_tour(base_url, runs):
        print(f"  - {label:<24} {url}")


def launch_showcase(root: Path, *, port: int = 8432, open_browser: bool = True, keep_existing: bool = False) -> dict[str, Any]:
    bundle = create_showcase_runs(root, keep_existing=keep_existing)
    chosen_port, port_note = resolve_server_port(port)
    if port_note:
        print(port_note)

    dashboard_port = bundle["runs"]["primary"].serve(port=chosen_port, open_browser=open_browser)
    base_url = f"http://127.0.0.1:{dashboard_port}"
    print_guided_tour(base_url, bundle["runs"])
    print(f"\nWorkspace: {root}")
    print(f"Dashboard: {base_url}")
    return {
        "store": bundle["store"],
        "runs": bundle["runs"],
        "workspace": root,
        "dashboard_url": base_url,
        "port": dashboard_port,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Build a multi-run GradGlass workspace that showcases every dashboard section."
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Workspace root for generated artifacts. Defaults to ./gg_workspace beside this example.",
    )
    parser.add_argument("--port", type=int, default=8432, help="Requested port for the GradGlass server.")
    parser.add_argument("--keep-existing", action="store_true", help="Reuse the latest showcase runs if they already exist in this workspace.")
    parser.set_defaults(open_browser=True)
    parser.add_argument("--open-browser", dest="open_browser", action="store_true", help="Open the dashboard in a browser.")
    parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="Do not open the dashboard in a browser.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    root = Path(args.root).resolve() if args.root else resolve_default_root(entrypoint=__file__)
    launch_showcase(root, port=args.port, open_browser=args.open_browser, keep_existing=args.keep_existing)


if __name__ == "__main__":
    main()
