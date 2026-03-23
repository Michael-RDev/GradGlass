"""
Pytest tests for gradglass.analysis.leakage using NumPy-only fixtures.
"""
from __future__ import annotations

import numpy as np

from gradglass.analysis.leakage import LeakageReport, run_leakage_detection

RND = 42


def _make_classification(n=600, n_features=10, seed=RND):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(n, n_features)).astype(np.float32)
    scales = rng.uniform(0.5, 3.0, size=(1, n_features)).astype(np.float32)
    offsets = rng.uniform(-4.0, 4.0, size=(1, n_features)).astype(np.float32)
    x = x * scales + offsets

    weights = rng.normal(0.0, 1.0, size=(n_features,)).astype(np.float32)
    logits = x @ weights + 0.3 * rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
    y = (logits > np.median(logits)).astype(np.int64)
    return x, y


def _train_test_split(x, y, test_size=0.3, seed=RND):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    split_idx = int(len(x) * (1.0 - test_size))
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def _fit_standardizer(x):
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def _apply_standardizer(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def _make_clean_split(n=600, n_features=10, test_size=0.3):
    """Return a clean, unscaled train/test split with no leakage."""
    x, y = _make_classification(n=n, n_features=n_features)
    return _train_test_split(x, y, test_size=test_size)


def _result(report: LeakageReport, check_id: str):
    """Extract a single LeakageCheckResult by check_id."""
    for r in report.results:
        if r.check_id == check_id:
            return r
    raise KeyError(f"check_id {check_id!r} not found in report")


class TestPreprocessingLeakage:
    def test_scaling_before_split_detected(self):
        """Scaling full data before split -> PREPROCESSING_LEAKAGE must FAIL."""
        x, y = _make_classification(n=600, n_features=10, seed=RND)
        mean, std = _fit_standardizer(x)
        x_scaled = _apply_standardizer(x, mean, std)
        x_tr, x_te, y_tr, y_te = _train_test_split(x_scaled, y, test_size=0.3, seed=RND)
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        r = _result(report, "PREPROCESSING_LEAKAGE")
        assert not r.passed, "Expected PREPROCESSING_LEAKAGE to FAIL when scaling was fit on full data"
        assert r.severity == "HIGH"

    def test_unscaled_split_passes(self):
        """Raw unscaled split -> check not applicable, must PASS."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "PREPROCESSING_LEAKAGE").passed

    def test_train_only_scaler_passes(self):
        """Scaler fit only on train, transform applied to both -> must PASS."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        mean, std = _fit_standardizer(x_tr)
        x_tr_s = _apply_standardizer(x_tr, mean, std)
        x_te_s = _apply_standardizer(x_te, mean, std)
        report = run_leakage_detection(x_tr_s, y_tr, x_te_s, y_te, verbose=False)
        assert _result(report, "PREPROCESSING_LEAKAGE").passed, (
            "Expected PREPROCESSING_LEAKAGE to PASS when scaler was fit only on train"
        )


class TestExactOverlap:
    def test_overlap_detected(self):
        """Verbatim train rows injected into test -> EXACT_OVERLAP must FAIL."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        x_te_leaky = np.vstack([x_te, x_tr[:20]])
        y_te_leaky = np.concatenate([y_te, y_tr[:20]])
        report = run_leakage_detection(x_tr, y_tr, x_te_leaky, y_te_leaky, verbose=False)
        r = _result(report, "EXACT_OVERLAP")
        assert not r.passed
        assert r.details["num_overlapping"] >= 20

    def test_no_overlap_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "EXACT_OVERLAP").passed


class TestTrainDuplicates:
    def test_duplicates_detected(self):
        """30 rows appended twice to training set -> TRAIN_DUPLICATES must FAIL."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        x_tr_dup = np.vstack([x_tr, x_tr[:30]])
        y_tr_dup = np.concatenate([y_tr, y_tr[:30]])
        report = run_leakage_detection(x_tr_dup, y_tr_dup, x_te, y_te, verbose=False)
        r = _result(report, "TRAIN_DUPLICATES")
        assert not r.passed
        assert r.details["total_extra_copies"] >= 30

    def test_no_train_duplicates_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "TRAIN_DUPLICATES").passed


class TestTestDuplicates:
    def test_duplicates_detected(self):
        """30 rows appended twice to test set -> TEST_DUPLICATES must FAIL."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        x_te_dup = np.vstack([x_te, x_te[:30]])
        y_te_dup = np.concatenate([y_te, y_te[:30]])
        report = run_leakage_detection(x_tr, y_tr, x_te_dup, y_te_dup, verbose=False)
        r = _result(report, "TEST_DUPLICATES")
        assert not r.passed
        assert r.details["total_extra_copies"] >= 30

    def test_no_test_duplicates_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "TEST_DUPLICATES").passed


class TestNearDuplicates:
    def test_near_duplicates_detected(self):
        """Train rows + tiny Gaussian noise in test -> NEAR_DUPLICATES must FAIL."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1e-6, size=(20, x_tr.shape[1])).astype(np.float32)
        near_copies = x_tr[:20].astype(np.float32) + noise
        x_te_leaky = np.vstack([x_te.astype(np.float32), near_copies])
        y_te_leaky = np.concatenate([y_te, y_tr[:20]])
        report = run_leakage_detection(x_tr, y_tr, x_te_leaky, y_te_leaky, verbose=False)
        assert not _result(report, "NEAR_DUPLICATES").passed

    def test_clean_data_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "NEAR_DUPLICATES").passed


class TestLabelDistribution:
    def test_skewed_distribution_detected(self):
        """All-class-0 train vs all-class-1 test -> LABEL_DISTRIBUTION must FAIL."""
        rng = np.random.default_rng(0)
        x_tr = rng.standard_normal((200, 5)).astype(np.float32)
        y_tr = np.zeros(200, dtype=np.int64)
        x_te = rng.standard_normal((60, 5)).astype(np.float32)
        y_te = np.ones(60, dtype=np.int64)
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert not _result(report, "LABEL_DISTRIBUTION").passed

    def test_balanced_distribution_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "LABEL_DISTRIBUTION").passed


class TestTargetCorrelation:
    def test_perfect_correlation_detected(self):
        """Feature column set equal to the target -> TARGET_CORRELATION must FAIL."""
        x, y = _make_classification(n=400, n_features=10, seed=RND)
        x[:, 0] = y.astype(np.float32)
        x_tr, x_te, y_tr, y_te = _train_test_split(x, y, test_size=0.3, seed=RND)
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        r = _result(report, "TARGET_CORRELATION")
        assert not r.passed
        assert r.details["num_suspicious"] >= 1

    def test_normal_correlation_passes(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert _result(report, "TARGET_CORRELATION").passed


class TestCleanDataFullReport:
    def test_all_checks_pass(self):
        """A cleanly split, unscaled dataset must pass every single check."""
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        failed = [r.check_id for r in report.results if not r.passed]
        assert report.passed, f"Checks unexpectedly failed: {failed}"
        assert report.num_failed == 0
        assert report.num_passed == len(report.results)

    def test_report_contains_all_check_ids(self):
        """Report must contain exactly the 8 expected check IDs."""
        expected = {
            "EXACT_OVERLAP",
            "TRAIN_DUPLICATES",
            "TEST_DUPLICATES",
            "NEAR_DUPLICATES",
            "LABEL_DISTRIBUTION",
            "FEATURE_STATS",
            "TARGET_CORRELATION",
            "PREPROCESSING_LEAKAGE",
        }
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert {r.check_id for r in report.results} == expected

    def test_returns_leakage_report_instance(self):
        x_tr, x_te, y_tr, y_te = _make_clean_split()
        report = run_leakage_detection(x_tr, y_tr, x_te, y_te, verbose=False)
        assert isinstance(report, LeakageReport)
