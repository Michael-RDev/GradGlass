from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


@dataclass
class LeakageCheckResult:
    check_id: str
    title: str
    passed: bool
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
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
    results: List[LeakageCheckResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "passed": self.passed,
            "num_passed": self.num_passed,
            "num_failed": self.num_failed,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_file(cls, path: Path) -> Optional["LeakageReport"]:
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        results = [LeakageCheckResult(**r) for r in data.get("results", [])]
        return cls(
            passed=data["passed"],
            num_passed=data["num_passed"],
            num_failed=data["num_failed"],
            total_duration_ms=data["total_duration_ms"],
            results=results,
        )


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
        self.max_samples = max_samples
        rng = np.random.RandomState(random_state)

        # Sub-sample if datasets are large
        n_train = len(train_x)
        n_test = len(test_x)

        if n_train > max_samples:
            idx = rng.choice(n_train, max_samples, replace=False)
            train_x = train_x[idx]
            train_y = train_y[idx]

        if n_test > max_samples:
            idx = rng.choice(n_test, max_samples, replace=False)
            test_x = test_x[idx]
            test_y = test_y[idx]

        # Flatten features to 2D: (N, D)
        self.train_x = train_x.reshape(len(train_x), -1).astype(np.float32)
        self.train_y = train_y.flatten()
        self.test_x = test_x.reshape(len(test_x), -1).astype(np.float32)
        self.test_y = test_y.flatten()

    def hash_rows(self, arr: np.ndarray) -> set:
        hashes = set()
        for row in arr:
            h = hashlib.md5(row.tobytes()).hexdigest()
            hashes.add(h)
        return hashes

    def check_exact_overlap(self) -> LeakageCheckResult:
        t0 = time.time()
        train_hashes = self.hash_rows(self.train_x)
        test_hashes = self.hash_rows(self.test_x)
        overlap = train_hashes & test_hashes
        n_overlap = len(overlap)
        passed = n_overlap == 0
        duration = (time.time() - t0) * 1000

        # Find indices of overlapping test samples
        overlap_indices = []
        if n_overlap > 0:
            test_hash_list = []
            for i, row in enumerate(self.test_x):
                h = hashlib.md5(row.tobytes()).hexdigest()
                if h in overlap:
                    overlap_indices.append(i)
                    if len(overlap_indices) >= 20:
                        break

        return LeakageCheckResult(
            check_id="EXACT_OVERLAP",
            title="Train/test exact-sample overlap",
            passed=passed,
            severity="CRITICAL",
            details={
                "num_overlapping": n_overlap,
                "overlap_test_indices": overlap_indices[:20],
                "train_size": len(self.train_x),
                "test_size": len(self.test_x),
            },
            recommendation=(
                f"{n_overlap} exact duplicates found between train and test. "
                "This will cause overestimated evaluation metrics."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_train_duplicates(self) -> LeakageCheckResult:
        t0 = time.time()
        hash_to_indices: dict[str, list[int]] = {}
        for i, row in enumerate(self.train_x):
            h = hashlib.md5(row.tobytes()).hexdigest()
            hash_to_indices.setdefault(h, []).append(i)

        dup_groups = {h: idxs for h, idxs in hash_to_indices.items() if len(idxs) > 1}
        total_extra = sum(len(v) - 1 for v in dup_groups.values())
        unique = len(hash_to_indices)
        passed = len(dup_groups) == 0
        duration = (time.time() - t0) * 1000

        sample_groups = []
        for h, idxs in list(dup_groups.items())[:5]:
            sample_groups.append(idxs[:5])

        return LeakageCheckResult(
            check_id="TRAIN_DUPLICATES",
            title="Duplicate samples within training set",
            passed=passed,
            severity="MEDIUM",
            details={
                "unique_samples": unique,
                "num_duplicate_groups": len(dup_groups),
                "total_extra_copies": total_extra,
                "sample_groups": sample_groups,
                "train_size": len(self.train_x),
            },
            recommendation=(
                f"{total_extra} extra duplicate copies in training set across "
                f"{len(dup_groups)} groups. May bias training."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_test_duplicates(self):
        t0 = time.time()
        hash_to_indices: dict[str, list[int]] = {}
        for i, row in enumerate(self.test_x):
            h = hashlib.md5(row.tobytes()).hexdigest()
            hash_to_indices.setdefault(h, []).append(i)

        dup_groups = {h: idxs for h, idxs in hash_to_indices.items() if len(idxs) > 1}
        total_extra = sum(len(v) - 1 for v in dup_groups.values())
        passed = len(dup_groups) == 0
        duration = (time.time() - t0) * 1000

        return LeakageCheckResult(
            check_id="TEST_DUPLICATES",
            title="Duplicate samples within test set",
            passed=passed,
            severity="MEDIUM",
            details={
                "num_duplicate_groups": len(dup_groups),
                "total_extra_copies": total_extra,
                "test_size": len(self.test_x),
            },
            recommendation=(
                f"{total_extra} duplicate copies in test set. May distort evaluation metrics." if not passed else ""
            ),
            duration_ms=duration,
        )

    def check_near_duplicates(self, threshold: float = 0.9999):
        t0 = time.time()

        # Normalize for cosine similarity
        train_norm = self.train_x / (np.linalg.norm(self.train_x, axis=1, keepdims=True) + 1e-10)
        test_norm = self.test_x / (np.linalg.norm(self.test_x, axis=1, keepdims=True) + 1e-10)

        near_dup_pairs = []
        chunk = 200
        for i in range(0, len(train_norm), chunk):
            sims = train_norm[i : i + chunk] @ test_norm.T
            hits = np.argwhere(sims > threshold)
            for ti, tei in hits:
                near_dup_pairs.append(
                    {"train_idx": int(i + ti), "test_idx": int(tei), "similarity": float(sims[ti, tei])}
                )
                if len(near_dup_pairs) >= 50:
                    break
            if len(near_dup_pairs) >= 50:
                break

        n_near = len(near_dup_pairs)
        passed = n_near == 0
        duration = (time.time() - t0) * 1000

        return LeakageCheckResult(
            check_id="NEAR_DUPLICATES",
            title="Train/test near-duplicate samples",
            passed=passed,
            severity="HIGH",
            details={
                "num_near_duplicates": n_near,
                "threshold": threshold,
                "train_sampled": len(self.train_x),
                "test_sampled": len(self.test_x),
                "pairs": near_dup_pairs[:20],
            },
            recommendation=(
                f"{n_near} near-duplicate pairs found (cosine sim > {threshold}). These may inflate evaluation metrics."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_label_distribution(self):
        t0 = time.time()
        train_counts = Counter(self.train_y.tolist())
        test_counts = Counter(self.test_y.tolist())
        all_labels = sorted(set(train_counts.keys()) | set(test_counts.keys()))

        train_total = sum(train_counts.values())
        test_total = sum(test_counts.values())

        train_dist = {str(l): round(train_counts.get(l, 0) / train_total, 4) for l in all_labels}
        test_dist = {str(l): round(test_counts.get(l, 0) / test_total, 4) for l in all_labels}

        per_class_diff = {}
        max_diff = 0.0
        for l in all_labels:
            diff = abs(train_dist[str(l)] - test_dist[str(l)])
            per_class_diff[str(l)] = round(diff, 4)
            max_diff = max(max_diff, diff)

        passed = max_diff < 0.15  # Allow up to 15% difference per class
        duration = (time.time() - t0) * 1000

        return LeakageCheckResult(
            check_id="LABEL_DISTRIBUTION",
            title="Train/test label-distribution consistency",
            passed=passed,
            severity="MEDIUM",
            details={
                "train_distribution": train_dist,
                "test_distribution": test_dist,
                "max_absolute_diff": round(max_diff, 4),
                "per_class_diff": per_class_diff,
            },
            recommendation=(
                f"Label distributions differ significantly (max diff = {max_diff:.4f}). Check data splitting strategy."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_feature_stats(self):
        t0 = time.time()

        # Per-feature statistics (axis=0) to preserve per-feature signal
        train_mean_per_feat = np.mean(self.train_x, axis=0)
        test_mean_per_feat = np.mean(self.test_x, axis=0)
        train_std_per_feat = np.std(self.train_x, axis=0)
        test_std_per_feat = np.std(self.test_x, axis=0)

        mean_diff = float(np.abs(train_mean_per_feat - test_mean_per_feat).mean())
        std_ratio = float(
            np.mean(
                np.maximum(train_std_per_feat, test_std_per_feat)
                / (np.minimum(train_std_per_feat, test_std_per_feat) + 1e-10)
            )
        )

        passed = mean_diff < 0.5 and std_ratio < 2.0
        duration = (time.time() - t0) * 1000

        # Top-10 most drifted features by mean diff
        feature_diffs = np.abs(train_mean_per_feat - test_mean_per_feat)
        top_indices = np.argsort(feature_diffs)[::-1][:10].tolist()
        top_drifted = [
            {"feature_idx": int(fi), "mean_diff": round(float(feature_diffs[fi]), 6)}
            for fi in top_indices
        ]

        return LeakageCheckResult(
            check_id="FEATURE_STATS",
            title="Train/test feature-statistics consistency",
            passed=passed,
            severity="MEDIUM",
            details={
                "mean_diff_avg": round(mean_diff, 6),
                "std_ratio_avg": round(std_ratio, 4),
                "num_features": int(self.train_x.shape[1]),
                "top_drifted_features": top_drifted,
            },
            recommendation=(
                f"Feature stats differ notably: mean_diff_avg={mean_diff:.4f}, std_ratio_avg={std_ratio:.4f}. "
                "This may indicate different preprocessing or data drift."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_preprocessing_leakage(self) -> LeakageCheckResult:
        """Detect scaler/normalizer fit on the full dataset before train/test split.

        Mathematical basis: if a zero-mean scaler (e.g. StandardScaler) is fit on
        the full dataset and the data is then split, the n-weighted combined
        per-feature mean of the two splits is *exactly* 0 by definition:

            (n_train * mean_train + n_test * mean_test) / N
            = (n_train * mean_train + n_test * mean_test) / N
            = mean_full  ≈  0

        When the scaler is fit only on training data, the combined weighted mean
        is (n_test / N) * (mean_test_raw - mean_train_raw) / std_train, which is
        nonzero for most real datasets.
        """
        t0 = time.time()
        n_train, n_test = len(self.train_x), len(self.test_x)

        train_mean = np.mean(self.train_x, axis=0)  # (D,)
        test_mean = np.mean(self.test_x, axis=0)    # (D,)
        train_std = np.std(self.train_x, axis=0)    # (D,)

        avg_train_mean_abs = float(np.abs(train_mean).mean())
        avg_train_std_diff = float(np.abs(train_std - 1.0).mean())
        avg_test_mean_abs = float(np.abs(test_mean).mean())

        # --- Step 1: check whether train data looks standardised ---
        train_looks_standardized = avg_train_mean_abs < 0.15 and avg_train_std_diff < 0.15

        if not train_looks_standardized:
            # Data does not appear to be standardised; check not applicable.
            return LeakageCheckResult(
                check_id="PREPROCESSING_LEAKAGE",
                title="Preprocessing / scaler leakage",
                passed=True,
                severity="HIGH",
                details={
                    "avg_train_mean_abs": round(avg_train_mean_abs, 4),
                    "avg_train_std_diff": round(avg_train_std_diff, 4),
                    "note": "Data does not appear standardised; check not applicable.",
                },
                recommendation="",
                duration_ms=(time.time() - t0) * 1000,
            )

        # --- Step 1b: if train mean is at floating-point precision the scaler
        # was fit ONLY on the training split (fit_transform(X_train) gives
        # mean=0 to ~1e-7).  Full-data scaling produces a train mean that is
        # small but measurably non-zero: (n_test/N)*(mean_tr_raw-mean_te_raw)/std_full.
        if avg_train_mean_abs < 1e-4:
            return LeakageCheckResult(
                check_id="PREPROCESSING_LEAKAGE",
                title="Preprocessing / scaler leakage",
                passed=True,
                severity="HIGH",
                details={
                    "avg_train_mean_abs": round(avg_train_mean_abs, 8),
                    "avg_train_std_diff": round(avg_train_std_diff, 4),
                    "note": "Train mean is at floating-point precision; scaler was fit on training data only.",
                },
                recommendation="",
                duration_ms=(time.time() - t0) * 1000,
            )

        # --- Step 2: compute n-weighted combined per-feature mean ---
        combined_mean = (n_train * train_mean + n_test * test_mean) / (n_train + n_test)
        avg_combined_abs = float(np.abs(combined_mean).mean())

        # Full-data scaling  → combined_mean ≈ 0  (mathematical identity)
        # Train-only scaling → combined_mean ≠ 0  (shifted by test/train discrepancy)
        leakage_detected = avg_test_mean_abs < 0.15 and avg_combined_abs < 0.05

        passed = not leakage_detected
        duration = (time.time() - t0) * 1000

        return LeakageCheckResult(
            check_id="PREPROCESSING_LEAKAGE",
            title="Preprocessing / scaler leakage",
            passed=passed,
            severity="HIGH",
            details={
                "avg_train_mean_abs": round(avg_train_mean_abs, 4),
                "avg_train_std_diff": round(avg_train_std_diff, 4),
                "avg_test_mean_abs": round(avg_test_mean_abs, 4),
                "avg_combined_mean_abs": round(avg_combined_abs, 6),
                "train_size": n_train,
                "test_size": n_test,
            },
            recommendation=(
                "Both train and test appear to have been scaled using the full dataset "
                "(n-weighted combined per-feature mean ≈ 0, both splits near zero-mean). "
                "Fit your scaler on training data only and use .transform() for test data."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def check_target_correlation(self, threshold: float = 0.95):
        t0 = time.time()

        # Combine train data for correlation
        all_x = np.concatenate([self.train_x, self.test_x], axis=0)
        all_y = np.concatenate([self.train_y, self.test_y], axis=0).astype(np.float32)

        n_features = all_x.shape[1]
        correlations = []
        suspicious = []

        # Sample features if too many
        feature_indices = list(range(n_features))
        if n_features > 1000:
            feature_indices = sorted(np.random.choice(n_features, 1000, replace=False).tolist())

        y_centered = all_y - np.mean(all_y)
        y_std = np.std(all_y) + 1e-10

        for fi in feature_indices:
            col = all_x[:, fi]
            col_centered = col - np.mean(col)
            col_std = np.std(col) + 1e-10
            corr = float(np.mean(col_centered * y_centered) / (col_std * y_std))
            correlations.append({"feature_idx": fi, "correlation": round(corr, 6)})
            if abs(corr) > threshold:
                suspicious.append({"feature_idx": fi, "correlation": round(corr, 6)})

        # Sort by absolute correlation descending
        correlations.sort(key=lambda c: abs(c["correlation"]), reverse=True)

        passed = len(suspicious) == 0
        duration = (time.time() - t0) * 1000

        return LeakageCheckResult(
            check_id="TARGET_CORRELATION",
            title="Feature-target correlation check",
            passed=passed,
            severity="HIGH",
            details={
                "num_suspicious": len(suspicious),
                "threshold": threshold,
                "suspicious_features": suspicious[:20],
                "top_correlations": correlations[:10],
            },
            recommendation=(
                f"{len(suspicious)} features have suspiciously high correlation with targets "
                f"(|r| > {threshold}). This may indicate label leakage in features."
                if not passed
                else ""
            ),
            duration_ms=duration,
        )

    def run_all(self):
        t0 = time.time()
        checks = [
            self.check_exact_overlap,
            self.check_train_duplicates,
            self.check_test_duplicates,
            self.check_near_duplicates,
            self.check_label_distribution,
            self.check_feature_stats,
            self.check_target_correlation,
            self.check_preprocessing_leakage,
        ]
        results = []
        for check_fn in checks:
            try:
                result = check_fn()
            except Exception as e:
                result = LeakageCheckResult(
                    check_id=check_fn.__name__.replace("check_", "").upper(),
                    title=f"Leakage check failed: {check_fn.__name__}",
                    passed=False,
                    severity="HIGH",
                    details={"error": str(e)},
                    recommendation="Check failed with an exception.",
                )
            results.append(result)

        total_duration = (time.time() - t0) * 1000
        num_passed = sum(1 for r in results if r.passed)
        num_failed = sum(1 for r in results if not r.passed)

        return LeakageReport(
            passed=num_failed == 0,
            num_passed=num_passed,
            num_failed=num_failed,
            total_duration_ms=total_duration,
            results=results,
        )


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
    detector = LeakageDetector(
        train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y,
        max_samples=max_samples, random_state=random_state,
    )
    report = detector.run_all()
    if save_path is not None:
        report.save(save_path)
    if verbose:
        _print_leakage_report(report)
    return report


def _print_leakage_report(report: LeakageReport) -> None:
    """Print a human-readable summary of a LeakageReport."""
    SEV_ICON = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    overall = "✅ PASSED" if report.passed else "❌ FAILED"
    print(f"\n{'─' * 55}")
    print(f"  GradGlass Leakage Report  │  {overall}")
    print(f"  {report.num_passed} passed · {report.num_failed} failed · {report.total_duration_ms:.0f} ms")
    print(f"{'─' * 55}")
    for r in report.results:
        icon = SEV_ICON.get(r.severity, "⚪")
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {icon} {r.title}")
        if not r.passed and r.recommendation:
            # Wrap recommendation to 70 chars
            words, line = r.recommendation.split(), ""
            for word in words:
                if len(line) + len(word) + 1 > 70:
                    print(f"         {line}")
                    line = word
                else:
                    line = (line + " " + word).strip()
            if line:
                print(f"         {line}")
    print(f"{'─' * 55}\n")
