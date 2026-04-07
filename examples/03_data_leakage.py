import torch
from torch.utils.data import DataLoader, TensorDataset
from gradglass import gg
from gradglass.analysis.leakage import run_leakage_detection
from _example_output import print_dashboard_next_steps, repo_workspace_root


def main():
    gg.configure(root=str(repo_workspace_root()), auto_open=False)

    # Generate some dummy data representing a potential leak overlap
    # We purposefully add 50 identical samples to both train and test to trigger a leakage warning.
    train_x = torch.randn(1000, 10)
    train_y = torch.randint(0, 2, (1000,))

    test_x = torch.randn(200, 10)
    test_y = torch.randint(0, 2, (200,))

    # Inject leakage (same features, same target)
    test_x[:50] = train_x[:50]
    test_y[:50] = train_y[:50]

    # --- Method 1: Using standalone detection (no gg.run needed) ---
    print("--- Running standalone leakage detection ---")
    standalone_report = run_leakage_detection(
        train_x=train_x.numpy(), train_y=train_y.numpy(), test_x=test_x.numpy(), test_y=test_y.numpy()
    )
    print(f"Leakage check passed? {standalone_report.passed}")

    # --- Method 2: Via DataLoaders wrapped in a Run ---
    print("\n--- Running DataLoader leakage detection via Run ---")
    run = gg.run(name="leakage_test_run")

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=32)

    run.check_leakage_from_loaders(
        train_loader=train_loader, test_loader=test_loader, max_samples=2000, print_summary=True
    )

    # You can also manually access raw arrays in `run.check_leakage()` directly!
    # run.check_leakage(train_x, train_y, test_x, test_y)

    print("Finished.")
    print_dashboard_next_steps(gg.store.root)


if __name__ == "__main__":
    main()
