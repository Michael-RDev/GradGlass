import time
from gradglass import gg

def main():
    gg.configure(auto_open=False)
    
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
    
    # 4. Demonstrate manual monitoring and cancellation
    print("\n--- Testing Run Interrupts and Monitoring ---")
    live_run = gg.run(name="live_monitor_test")
    
    print("Starting background dashboard server...")
    port = live_run.serve(port=8432, open_browser=False)
    print(f"Dashboard serving at http://localhost:{port}")
    
    # Simulate a little training
    live_run.log(loss=1.0)
    time.sleep(1)
    
    # Abruptly cancel the run
    print("Canceling run due to user interrupt...")
    live_run.cancel(reason="manual stop requested via external trigger", open=False)

    print("Complete.")

    # 5. gg.open_last()
    # In a real environment, uncommenting this will pop open a browser window to the latest run
    # gg.open_last()

if __name__ == "__main__":
    main()
