import time
from gradglass import gg

from _example_output import print_dashboard_next_steps, repo_workspace_root


def main():
    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    # 1. Create a dummy run just to populate the workspace
    dummy_run = gg.run(name="dummy_for_workspace")
    dummy_run.log(loss=0.1)
    dummy_run.finish(open=False, analyze=False)

    # 2. List all past runs in the workspace
    print("--- Listing Past Runs ---")
    runs = gg.list_runs()
    for meta in runs:
        print(f"Run: {meta.get('name')} (ID: {meta.get('run_id')}) | Status: {meta.get('status')}")
        print(f"  Storage: {meta.get('storage_mb')} MB | Latest Loss: {meta.get('latest_loss')}")

    if not runs:
        return

    # 3. Retrieve a specific run and extract data
    last_run_id = runs[-1]["run_id"]
    print(f"\n--- Loading Run: {last_run_id} ---")
    retrieved_run = gg.get_run(last_run_id)
    print(f"Retrieved Run Metadata Config: {retrieved_run.options}")

    # 4. Demonstrate the recommended post-run workflow
    print("\n--- Training First, Serving Afterward ---")
    follow_up_run = gg.run(name="serve_after_training_demo")
    follow_up_run.log(loss=1.0)
    time.sleep(1)
    follow_up_run.finish(open=False, analyze=False)

    print("Run complete.")
    print_dashboard_next_steps(gg.store.root, live_monitor=True)

    print("Complete.")

if __name__ == "__main__":
    main()
