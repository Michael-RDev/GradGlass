from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_workspace_root() -> Path:
    return repo_root() / "gg_workspace"


def uses_repo_workspace(workspace_root: Path | str) -> bool:
    return Path(workspace_root).resolve() == repo_workspace_root().resolve()


def serve_command_for_workspace(workspace_root: Path | str, *, port: int = 8432) -> str:
    root = Path(workspace_root).resolve()
    if uses_repo_workspace(root):
        return f"gradglass serve --port {port}"
    return f"GRADGLASS_ROOT='{root}' gradglass serve --port {port}"


def print_dashboard_next_steps(
    workspace_root: Path | str,
    *,
    port: int = 8432,
    live_monitor: bool = False,
    label: str = "Workspace",
) -> None:
    root = Path(workspace_root).resolve()
    print(f"{label}: {root}")
    print(f"Start dashboard with: {serve_command_for_workspace(root, port=port)}")
    print("If port 8432 is already occupied, stop the old server with: gradglass stop --port 8432")
    if live_monitor:
        print("This example can also use live monitoring with monitor=True during training.")
