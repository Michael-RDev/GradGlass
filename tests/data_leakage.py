"""
Pytest tests for gradglass.analysis.leakage (numpy / sklearn API).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gradglass.analysis.leakage import LeakageDetector, LeakageReport, run_leakage_detection

RND = 42


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_clean_split(n=600, n_features=10, test_size=0.3):
    """Return a clean, unscaled train/test split with no leakage."""
    X, y = make_classification(
        n_samples=n, n_features=n_features, n_informative=4,
        n_redundant=2, class_sep=1.0, flip_y=0.05, random_state=RND,
    )
    return train_test_split(X, y, test_size=test_size, random_state=RND)


def _result(report: LeakageReport, check_id: str):
    """Extract a single LeakageCheckResult by check_id."""
    for r in report.results:
        if r.check_id == check_id:
            return r
    raise KeyError(f"check_id {check_id!r} not found in report")



# ---------------------------------------------------------------------------
# 1. Preprocessing / scaler leakage
# ---------------------------------------------------------------------------

class TestPreprocessingLeakage:
    def test_scaling_before_split_detected(self):
        """fit_transform on full data -> PREPROCESSING_LEAKAGE must FAIL."""
        X, y = make_classification(n_samples=600, n_features=10, n_informative=4, random_state=RND)
        X_scaled = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.3, random_state=RND)
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        r = _result(report, "PREPROCESSING_LEAKAGE")
        assert not r.passed, "Expected PREPROCESSING_LEAKAGE to FAIL when scaler was fit on full data"
        assert r.severity == "HIGH"

    def test_unscaled_split_passes(self):
        """Raw unscaled split -> check not applicable, must PASS."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "PREPROCESSING_LEAKAGE").passed

    def test_train_only_scaler_passes(self):
        """Scaler fit only on train, transform applied to both -> must PASS."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        report = run_leakage_detection(X_tr_s, y_tr, X_te_s, y_te, verbose=False)
        assert _result(report, "PREPROCESSING_LEAKAGE").passed, (
            "Expected PREPROCESSING_LEAKAGE to PASS when scaler was fit only on train"
        )


# ---------------------------------------------------------------------------
# 2. Exact overlap
# ---------------------------------------------------------------------------

class TestExactOverlap:
    def test_overlap_detected(self):
        """Verbatim train rows injected into test -> EXACT_OVERLAP must FAIL."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        X_te_leaky = np.vstack([X_te, X_tr[:20]])
        y_te_leaky = np.concatenate([y_te, y_tr[:20]])
        report = run_leakage_detection(X_tr, y_tr, X_te_leaky, y_te_leaky, verbose=False)
        r = _result(report, "EXACT_OVERLAP")
        assert not r.passed
        assert r.details["num_overlapping"] >= 20

    def test_no_overlap_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "EXACT_OVERLAP").passed


# ---------------------------------------------------------------------------
# 3. Train-set duplicates
# ---------------------------------------------------------------------------

class TestTrainDuplicates:
    def test_duplicates_detected(self):
        """30 rows appended twice to training set -> TRAIN_DUPLICATES must FAIL."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        X_tr_dup = np.vstack([X_tr, X_tr[:30]])
        y_tr_dup = np.concatenate([y_tr, y_tr[:30]])
        report = run_leakage_detection(X_tr_dup, y_tr_dup, X_te, y_te, verbose=False)
        r = _result(report, "TRAIN_DUPLICATES")
        assert not r.passed
        assert r.details["total_extra_copies"] >= 30

    def test_no_train_duplicates_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "TRAIN_DUPLICATES").passed


# ---------------------------------------------------------------------------
# 4. Test-set duplicates
# ---------------------------------------------------------------------------

class TestTestDuplicates:
    def test_duplicates_detected(self):
        """30 rows appended twice to test set -> TEST_DUPLICATES must FAIL."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        X_te_dup = np.vstack([X_te, X_te[:30]])
        y_te_dup = np.concatenate([y_te, y_te[:30]])
        report = run_leakage_detection(X_tr, y_tr, X_te_dup, y_te_dup, verbose=False)
        r = _result(report, "TEST_DUPLICATES")
        assert not r.passed
        assert r.details["total_extra_copies"] >= 30

    def test_no_test_duplicates_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "TEST_DUPLICATES").passed


# ---------------------------------------------------------------------------
# 5. Near duplicates
# ---------------------------------------------------------------------------

class TestNearDuplicates:
    def test_near_duplicates_detected(self):
        """Train rows + 1e-6 Gaussian noise injected into test -> NEAR_DUPLICATES must FAIL."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1e-6, size=(20, X_tr.shape[1])).astype(np.float32)
        near_copies = X_tr[:20].astype(np.float32) + noise
        X_te_leaky = np.vstack([X_te.astype(np.float32), near_copies])
        y_te_leaky = np.concatenate([y_te, y_tr[:20]])
        report = run_leakage_detection(X_tr, y_tr, X_te_leaky, y_te_leaky, verbose=False)
        assert not _result(report, "NEAR_DUPLICATES").passed

    def test_clean_data_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "NEAR_DUPLICATES").passed


# ---------------------------------------------------------------------------
# 6. Label distribution
# ---------------------------------------------------------------------------

class TestLabelDistribution:
    def test_skewed_distribution_detected(self):
        """All-class-0 train vs all-class-1 test -> LABEL_DISTRIBUTION must FAIL."""
        rng = np.random.default_rng(0)
        X_tr = rng.standard_normal((200, 5)).astype(np.float32)
        y_tr = np.zeros(200, dtype=np.int64)
        X_te = rng.standard_normal((60, 5)).astype(np.float32)
        y_te = np.ones(60, dtype=np.int64)
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert not _result(report, "LABEL_DISTRIBUTION").passed

    def test_balanced_distribution_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "LABEL_DISTRIBUTION").passed


# ---------------------------------------------------------------------------
# 7. Target-correlation (label leakage in features)
# ---------------------------------------------------------------------------

class TestTargetCorrelation:
    def test_perfect_correlation_detected(self):
        """Feature column set equal to the target -> TARGET_CORRELATION must FAIL."""
        X, y = make_classification(n_samples=400, n_features=10, random_state=RND)
        X[:, 0] = y.astype(float)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=RND)
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        r = _result(report, "TARGET_CORRELATION")
        assert not r.passed
        assert r.details["num_suspicious"] >= 1

    def test_normal_correlation_passes(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert _result(report, "TARGET_CORRELATION").passed


# ---------------------------------------------------------------------------
# 8. Clean data - full report
# ---------------------------------------------------------------------------

class TestCleanDataFullReport:
    def test_all_checks_pass(self):
        """A cleanly split, unscaled dataset must pass every single check."""
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        failed = [r.check_id for r in report.results if not r.passed]
        assert report.passed, f"Checks unexpectedly failed: {failed}"
        assert report.num_failed == 0
        assert report.num_passed == len(report.results)

    def test_report_contains_all_check_ids(self):
        """Report must contain exactly the 8 expected check IDs."""
        expected = {
            "EXACT_OVERLAP", "TRAIN_DUPLICATES", "TEST_DUPLICATES",
            "NEAR_DUPLICATES", "LABEL_DISTRIBUTION", "FEATURE_STATS",
            "TARGET_CORRELATION", "PREPROCESSING_LEAKAGE",
        }
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert {r.check_id for r in report.results} == expected

    def test_returns_leakage_report_instance(self):
        X_tr, X_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(X_tr, y_tr, X_te, y_te, verbose=False)
        assert isinstance(report, LeakageReport)
