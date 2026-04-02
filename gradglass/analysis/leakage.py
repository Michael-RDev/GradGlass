from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from gradglass.analysis.data_monitor.builder import DatasetMonitorBuilder
from gradglass.analysis.data_monitor.models import CheckStatus, DatasetMonitorReport, PipelineStage, TaskType


LEGACY_CHECK_IDS = {
    "Train/test exact-sample overlap": "EXACT_OVERLAP",
    "Duplicate samples within training set": "TRAIN_DUPLICATES",
    "Duplicate samples within test set": "TEST_DUPLICATES",
    "Train/test near-duplicate samples": "NEAR_DUPLICATES",
    "Train/test label-distribution consistency": "LABEL_DISTRIBUTION",
    "Train/test feature-statistics consistency": "FEATURE_STATS",
    "Feature-target correlation check": "TARGET_CORRELATION",
    "Preprocessing / scaler leakage": "PREPROCESSING_LEAKAGE",
}


@dataclass
class LeakageCheckResult:
    check_id: str
    title: str
    passed: bool
    severity: str
    details: dict = field(default_factory=dict)
    recommendation: str = ""
    duration_ms: float = 0.0

    def to_dict(self):
        return {
            "check_id": self.check_id,
            "title": self.title,
            "passed": self.passed,
            "severity": self.severity,
            "details": self.details,
            "recommendation": self.recommendation,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class LeakageReport:
    passed: bool
    num_passed: int
    num_failed: int
    total_duration_ms: float
    results: list[LeakageCheckResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "passed": self.passed,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "results": [result.to_dict() for result in self.results],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=4)

    @classmethod
    def from_file(cls, path: Path) -> Optional["LeakageReport"]:
        if not path.exists():
            return None
        with open(path) as handle:
            data = json.load(handle)
        results = [LeakageCheckResult(**result) for result in data.get("results", [])]
        return cls(
            passed=data.get("passed", False),
            num_passed=data.get("num_passed", 0),
            num_failed=data.get("num_failed", 0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            results=results,
        )


def _coerce_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and callable(value.numpy):
        try:
            return value.numpy()
        except Exception:
            return np.asarray(value)
    return np.asarray(value)


def _subset_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    max_samples: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    if len(train_x) > max_samples:
        idx = rng.choice(len(train_x), max_samples, replace=False)
        train_x = train_x[idx]
        train_y = train_y[idx]
    if len(test_x) > max_samples:
        idx = rng.choice(len(test_x), max_samples, replace=False)
        test_x = test_x[idx]
        test_y = test_y[idx]
    return train_x, train_y, test_x, test_y


def _infer_task_from_labels(labels: np.ndarray) -> TaskType:
    labels = np.asarray(labels)
    if labels.size == 0:
        return TaskType.UNKNOWN
    flat = labels.reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        return TaskType.CLASSIFICATION
    unique_count = len(np.unique(flat))
    if unique_count <= min(20, max(2, flat.size // 20)):
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION


def _infer_run_dir_from_save_path(save_path: Path | None) -> Path | None:
    if save_path is None:
        return None
    if save_path.parent.name == "analysis":
        return save_path.parent.parent
    return None


def project_monitor_report_to_legacy(report: DatasetMonitorReport) -> LeakageReport:
    results = []
    for check in report.checks:
        if check.category.value != "leakage":
            continue
        details = dict(check.payload)
        if check.evidence:
            details.setdefault("evidence", check.evidence)
        if check.metrics:
            for key, value in check.metrics.items():
                details.setdefault(key, value)
        results.append(
            LeakageCheckResult(
                check_id=LEGACY_CHECK_IDS.get(check.name, check.name.upper().replace("/", "_").replace(" ", "_")),
                title=check.name,
                passed=check.status == CheckStatus.PASSED,
                severity=check.severity.value,
                details=details,
                recommendation=check.recommendation or "",
                duration_ms=check.duration_ms,
            )
        )
    num_passed = sum(1 for result in results if result.passed)
    num_failed = sum(1 for result in results if not result.passed)
    total_duration_ms = sum(result.duration_ms for result in results)
    return LeakageReport(
        passed=num_failed == 0,
        num_passed=num_passed,
        num_failed=num_failed,
        total_duration_ms=total_duration_ms,
        results=results,
    )


def project_monitor_report_to_legacy_dict(report: DatasetMonitorReport) -> dict:
    return project_monitor_report_to_legacy(report).to_dict()


def _save_standalone_monitor_report(report: DatasetMonitorReport, save_path: Path) -> None:
    monitor_path = save_path.with_name("dataset_monitor.json")
    summary_path = save_path.with_name("dataset_monitor_summary.txt")
    with open(monitor_path, "w") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)
    with open(summary_path, "w") as handle:
        handle.write(report.summary_text)


def build_monitor_report_for_arrays(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    max_samples: int = 2000,
    random_state: int | None = None,
    dataset_name: str | None = None,
    run_dir: Path | None = None,
    run_id: str | None = None,
    save: bool = True,
) -> DatasetMonitorReport:
    train_x = _coerce_numpy(train_x)
    train_y = _coerce_numpy(train_y)
    test_x = _coerce_numpy(test_x)
    test_y = _coerce_numpy(test_y)
    train_x, train_y, test_x, test_y = _subset_arrays(train_x, train_y, test_x, test_y, max_samples, random_state)
    builder = DatasetMonitorBuilder(
        task=_infer_task_from_labels(train_y),
        dataset_name=dataset_name or "Leakage Detection Dataset",
        task_hint="leakage_detection",
        config={"sample_budget_per_split": max_samples, "random_seed": random_state or 0},
        run_dir=run_dir,
        run_id=run_id,
    )
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="train",
        data=train_x,
        labels=train_y,
        metadata={"source": "legacy_leakage_wrapper"},
    )
    builder.record_stage(
        PipelineStage.SPLITTING,
        split="test",
        data=test_x,
        labels=test_y,
        metadata={"source": "legacy_leakage_wrapper"},
    )
    return builder.finalize(save=save)


class LeakageDetector:
    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        max_samples: int = 2000,
        random_state: Optional[int] = None,
    ):
        self.train_x = _coerce_numpy(train_x)
        self.train_y = _coerce_numpy(train_y)
        self.test_x = _coerce_numpy(test_x)
        self.test_y = _coerce_numpy(test_y)
        self.max_samples = max_samples
        self.random_state = random_state

    def run_all(self) -> LeakageReport:
        monitor_report = build_monitor_report_for_arrays(
            self.train_x,
            self.train_y,
            self.test_x,
            self.test_y,
            max_samples=self.max_samples,
            random_state=self.random_state,
            save=False,
        )
        return project_monitor_report_to_legacy(monitor_report)


def run_leakage_detection(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    max_samples: int = 2000,
    save_path: Optional[Path] = None,
    verbose: bool = True,
    random_state: Optional[int] = None,
) -> LeakageReport:
    run_dir = _infer_run_dir_from_save_path(save_path)
    monitor_report = build_monitor_report_for_arrays(
        train_x,
        train_y,
        test_x,
        test_y,
        max_samples=max_samples,
        random_state=random_state,
        run_dir=run_dir,
        save=run_dir is not None,
    )
    if save_path is not None and run_dir is None:
        _save_standalone_monitor_report(monitor_report, save_path)
    report = project_monitor_report_to_legacy(monitor_report)
    if save_path is not None:
        report.save(save_path)
    if verbose:
        _print_leakage_report(report)
    return report


def _print_leakage_report(report: LeakageReport) -> None:
    severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "⚪"}
    overall = "✅ PASSED" if report.passed else "❌ FAILED"
    print(f"\n{'─' * 55}")
    print(f"  GradGlass Leakage Report  │  {overall}")
    print(f"  {report.num_passed} passed · {report.num_failed} failed · {report.total_duration_ms:.0f} ms")
    print(f"{'─' * 55}")
    for result in report.results:
        icon = severity_icon.get(result.severity, "⚪")
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {icon} {result.title}")
        if not result.passed and result.recommendation:
            print(f"         {result.recommendation}")
    print(f"{'─' * 55}\n")
