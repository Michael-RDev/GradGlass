import numpy as np
from gradglass import gg
from gradglass.analysis.data_monitor import DatasetMonitorConfig

def main():
    gg.configure(auto_open=False)
    
    # We can create a strict configuration for the data monitor constraints
    config = DatasetMonitorConfig(
        max_nan_ratio=0.01,
        max_zero_ratio=0.1,
        track_ranges=True,
    )
    
    # Instantiate the data monitor
    # It can be accessed generically through gg.monitor_dataset or run.monitor_dataset
    monitor = gg.monitor_dataset(
        task="classification",
        dataset_name="dummy_pipeline",
        task_hint="tabular",
        config=config
    )
    
    # 1. Simulate Raw Data Loader
    monitor.register_stage("Raw_Ingest", feature_names=["age", "income", "credit_score"])
    
    raw_data = np.array([
        [25, 45000, 710],
        [np.nan, 50000, 680],  # Inject a NaN to see tracking behavior
        [42, 120000, 810]
    ])
    raw_labels = np.array([0, 1, 1])
    
    monitor.record_batch(stage="Raw_Ingest", batch=raw_data, labels=raw_labels)

    # 2. Simulate Preprocessing / Imputation
    monitor.register_stage("Imputed", feature_names=["age", "income", "credit_score"])
    
    # (Mock imputation of NaN -> 35 mean replacement)
    imputed_data = np.array([
        [25, 45000, 710],
        [35, 50000, 680], 
        [42, 120000, 810]
    ])
    
    monitor.record_batch(stage="Imputed", batch=imputed_data, labels=raw_labels)

    # 3. Simulate Normalization
    monitor.register_stage("Normalized", feature_names=["age", "income", "credit_score"])
    
    # Mock norm
    norm_data = (imputed_data - imputed_data.mean(axis=0)) / (imputed_data.std(axis=0) + 1e-8)
    # Record sample by sample
    for i in range(len(norm_data)):
        monitor.record(stage="Normalized", sample=norm_data[i], label=raw_labels[i])

    # End the pipeline monitoring
    report = monitor.build_report()
    monitor.close()
    
    # Display the collected insight
    print(f"\nFinal Analysis Report for dataset: {report.dataset_name}")
    print(f"Total Stages captured: {len(report.stages)}")
    for stage_meta in report.stages:
        print(f" > Stage '{stage_meta.name}': {stage_meta.sample_count} rows. Issue Count: {len(stage_meta.issues)}")
        for issue in stage_meta.issues:
            print(f"   - [Warning] {issue['type']} on feature {issue.get('feature_index', '?')}: {issue['severity']}")

if __name__ == "__main__":
    main()
