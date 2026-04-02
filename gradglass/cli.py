from __future__ import annotations
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="gradglass", description="GradGlass — Neural Network Transparency Engine")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument("--port", type=int, default=8432, help="Port to bind to")
    serve_parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    subparsers.add_parser("list", help="List all runs")
    open_parser = subparsers.add_parser("open", help="Open a run in the dashboard")
    open_parser.add_argument("run_id", nargs="?", help="Run ID to open (latest if omitted)")
    monitor_parser = subparsers.add_parser(
        "monitor", help="Start live monitoring dashboard (before or during training)"
    )
    monitor_parser.add_argument("--port", type=int, default=8432, help="Port to bind to")
    monitor_parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    stop_parser = subparsers.add_parser("stop", help="Stop a detached GradGlass monitor server")
    stop_parser.add_argument("run_id", nargs="?", help="Run ID whose detached monitor should be stopped")
    stop_parser.add_argument("--port", type=int, default=None, help="Stop the GradGlass server listening on this port")
    stop_parser.add_argument("--all", action="store_true", help="Stop all detached run monitor servers")
    analyze_parser = subparsers.add_parser("analyze", help="Run post-training analysis on a run")
    analyze_parser.add_argument("run_id", nargs="?", help="Run ID to analyze (latest if omitted)")
    analyze_parser.add_argument("--open", action="store_true", help="Open dashboard after analysis")
    analyze_parser.add_argument("--tests", nargs="*", default=None, help="Specific test IDs to run (default: all)")
    args = parser.parse_args()
    if args.command == "serve":
        from gradglass.core import gg
        from gradglass.server import create_app, start_server_blocking

        app = create_app(gg.store)
        start_server_blocking(app, port=args.port, open_browser=not args.no_browser)
    elif args.command == "list":
        from gradglass.core import gg

        runs = gg.list_runs()
        if not runs:
            print("No runs found.")
            return
        print(f"\n{'Name':<30} {'Steps':>8} {'Loss':>10} {'Status':<12} {'Storage':>10}")
        print("-" * 75)
        for r in runs:
            name = r.get("name", "?")[:29]
            steps = r.get("total_steps", "?")
            loss = r.get("latest_loss")
            loss_str = f"{loss:.4f}" if loss is not None else "—"
            status = r.get("status", "?")
            storage = f"{r.get('storage_mb', 0)} MB"
            print(f"{name:<30} {steps:>8} {loss_str:>10} {status:<12} {storage:>10}")
        print()
    elif args.command == "open":
        from gradglass.core import gg

        if args.run_id:
            run = gg.get_run(args.run_id)
            run.open()
        else:
            gg.open_last()
    elif args.command == "monitor":
        from gradglass.core import gg
        from gradglass.server import create_app, start_server_blocking

        app = create_app(gg.store)
        start_server_blocking(app, port=args.port, open_browser=not args.no_browser)
    elif args.command == "stop":
        from gradglass.core import gg
        from gradglass.monitor_control import stop_gradglass_monitor

        if args.all and args.run_id:
            parser.error("Use --all without a run_id.")
        if args.all and args.port is not None:
            parser.error("Use --all without --port.")
        if not args.all and args.port is None and not args.run_id:
            parser.error("Specify a run_id, --port, or --all.")

        results = stop_gradglass_monitor(gg.store, run_id=args.run_id, port=args.port, stop_all=args.all)
        exit_code = 0
        for result in results:
            print(result.message)
            if result.status in {"refused", "error", "not_found", "usage_error"}:
                exit_code = 1
        if exit_code:
            raise SystemExit(exit_code)
    elif args.command == "analyze":
        from gradglass.core import gg

        if args.run_id:
            run = gg.get_run(args.run_id)
        else:
            runs = gg.list_runs()
            if not runs:
                print("No runs found.")
                return
            runs.sort(key=lambda r: r.get("start_time", ""), reverse=True)
            run = gg.get_run(runs[0]["run_id"])
        tests = args.tests if args.tests else "all"
        report = run.analyze(tests=tests)
        if args.open:
            run.open()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
