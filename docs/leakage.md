# Data Leakage Detection

GradGlass includes a built-in leakage detection engine that runs 8 statistical checks across your train and test splits. Results are saved to `analysis/leakage_report.json` inside the run directory and rendered on the **Leakage Report** page of the dashboard.

---

## Table of Contents

- [Quick Usage](#quick-usage)
- [The 8 Checks](#the-8-checks)
- [Severity Reference](#severity-reference)
- [`LeakageDetector` Class](#leakagedetector-class)
- [`leakage_report.json` Schema](#leakage_reportjson-schema)

---

## Quick Usage

### With numpy arrays

```python
run.check_leakage(X_train, y_train, X_test, y_test)
```

### With PyTorch `DataLoader` objects

```python
run.check_leakage_from_loaders(train_loader, test_loader)
```

Both methods save results to `analysis/leakage_report.json` and return a `LeakageReport` object.

### Standalone (without a run)

```python
from gradglass import LeakageDetector

detector = LeakageDetector(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test, max_samples=2000)
report = detector.run_all()

for result in report.results:
    print(result.check_id, result.passed, result.details)
```

---

## The 8 Checks

### 1. `EXACT_OVERLAP` — Exact Train/Test Duplicate Detection

**Severity:** CRITICAL

MD5-hashes every sample in both sets and finds exact row-level duplicates between train and test. Even a single overlapping sample is a critical leakage issue for evaluation purposes.

| Result | Condition |
|--------|-----------|
| PASS | Zero exact matches found |
| FAIL | One or more train samples found verbatim in the test set |

**Details returned:** `overlap_count`, `overlap_rate`, `example_indices`

---

### 2. `TRAIN_DUPLICATES` — Duplicate Samples Within Training Set

**Severity:** MEDIUM

Checks for rows that appear more than once within the training set itself. Duplicates inflate effective sample counts and can bias learning toward repeated patterns.

| Result | Condition |
|--------|-----------|
| PASS | No duplicates found |
| WARN | Duplicates exist (reports count and rate) |

**Details returned:** `duplicate_count`, `duplicate_rate`

---

### 3. `TEST_DUPLICATES` — Duplicate Samples Within Test Set

**Severity:** MEDIUM

Same as `TRAIN_DUPLICATES` but applied to the test set. Duplicate test samples inflate the apparent importance of certain samples in your evaluation metrics.

| Result | Condition |
|--------|-----------|
| PASS | No duplicates found |
| WARN | Duplicates exist (reports count and rate) |

**Details returned:** `duplicate_count`, `duplicate_rate`

---

### 4. `NEAR_DUPLICATES` — Near-Duplicate Detection

**Severity:** HIGH

Computes pairwise cosine similarity between every train sample and every test sample (chunked matrix multiply to keep memory bounded). Flags pairs that exceed the similarity threshold.

| Result | Condition |
|--------|-----------|
| PASS | No pair exceeds the threshold |
| WARN | Near-duplicate pairs found |

**Default threshold:** `0.99` cosine similarity

**Details returned:** `near_duplicate_count`, `near_duplicate_rate`, `threshold`

> **Note:** On large datasets this check uses `max_samples` subsampling (default 2,000) to remain tractable.

---

### 5. `LABEL_DISTRIBUTION` — Label Distribution Mismatch

**Severity:** MEDIUM

Computes the per-class frequency in train vs test and flags any class where the absolute distribution difference exceeds 15%.

| Result | Condition |
|--------|-----------|
| PASS | All per-class differences ≤ 15% |
| WARN | At least one class differs by > 15% |

**Details returned:** `max_class_diff`, `distribution_diff` (per-class dict)

---

### 6. `FEATURE_STATS` — Feature Statistics Mismatch

**Severity:** MEDIUM

Compares per-feature mean and standard deviation between train and test. Large statistical differences can indicate that the test set comes from a different distribution — a form of covariate shift that can make evaluation metrics misleading.

| Result | Condition |
|--------|-----------|
| PASS | All feature means differ by ≤ 0.1 AND all std ratios are within 1.5× |
| WARN | Any feature mean diff > 0.1 OR any std ratio > 1.5 |

**Details returned:** `max_mean_diff`, `max_std_ratio`, `flagged_features`

---

### 7. `TARGET_CORRELATION` — Target/Label Leakage via Feature Correlation

**Severity:** HIGH

Computes Pearson correlation between each feature and the target label on the training set. Features with |r| > 0.95 are almost certainly leaking label information — they are either derived from the label or encode it directly.

| Result | Condition |
|--------|-----------|
| PASS | No feature exceeds the correlation threshold |
| FAIL | At least one feature has \|r\| > 0.95 with the target |

**Default threshold:** `0.95`

**Details returned:** `high_correlation_features` (list of `{feature_index, correlation}` dicts)

> **Note:** This check only runs for classification and regression tasks where `y_train` is a 1-D label array.

---

## Severity Reference

| Check ID | Severity | Max Status |
|----------|----------|------------|
| `EXACT_OVERLAP` | CRITICAL | FAIL |
| `NEAR_DUPLICATES` | HIGH | FAIL |
| `TARGET_CORRELATION` | HIGH | FAIL |
| `PREPROCESSING_LEAKAGE` | HIGH | FAIL |
| `TRAIN_DUPLICATES` | MEDIUM | FAIL |
| `TEST_DUPLICATES` | MEDIUM | FAIL |
| `LABEL_DISTRIBUTION` | MEDIUM | FAIL |
| `FEATURE_STATS` | MEDIUM | FAIL |

---

## `LeakageDetector` Class

```python
from gradglass import LeakageDetector

detector = LeakageDetector(
    train_x=X_train,
    train_y=y_train,
    test_x=X_test,
    test_y=y_test,
    max_samples=2000,
    random_state=42,   # optional — makes subsampling reproducible
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_x` | `np.ndarray` | — | Training features |
| `train_y` | `np.ndarray` | — | Training labels |
| `test_x` | `np.ndarray` | — | Test features |
| `test_y` | `np.ndarray` | — | Test labels |
| `max_samples` | `int` | `2000` | Subsample size — if train or test has more rows, a random subset is drawn for all checks |
| `random_state` | `int \| None` | `None` | Seed for reproducible subsampling |

### Methods

#### `detector.run_all() -> LeakageReport`

Run all 8 checks and return a `LeakageReport`.

#### Individual check methods

Each method returns a `LeakageCheckResult`:

| Method | Check ID |
|--------|----------|
| `detector.check_exact_overlap()` | `EXACT_OVERLAP` |
| `detector.check_train_duplicates()` | `TRAIN_DUPLICATES` |
| `detector.check_test_duplicates()` | `TEST_DUPLICATES` |
| `detector.check_near_duplicates()` | `NEAR_DUPLICATES` |
| `detector.check_label_distribution()` | `LABEL_DISTRIBUTION` |
| `detector.check_feature_stats()` | `FEATURE_STATS` |
| `detector.check_target_correlation()` | `TARGET_CORRELATION` |
| `detector.check_preprocessing_leakage()` | `PREPROCESSING_LEAKAGE` |

### `LeakageCheckResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `check_id` | `str` | One of the 8 check IDs above |
| `title` | `str` | Human-readable check name |
| `passed` | `bool` | `True` if the check found no issues |
| `severity` | `str` | `"LOW"`, `"MEDIUM"`, `"HIGH"`, or `"CRITICAL"` |
| `details` | `dict` | Check-specific diagnostic values |
| `recommendation` | `str` | Actionable advice (non-empty on failure) |
| `duration_ms` | `float` | Wall-clock time this check took |

### `LeakageReport` Fields

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | `True` when all checks passed |
| `num_passed` | `int` | Number of checks that passed |
| `num_failed` | `int` | Number of checks that failed |
| `total_duration_ms` | `float` | Total wall-clock time for all checks |
| `results` | `list[LeakageCheckResult]` | All 8 check results in run order |

### Convenience function

```python
from gradglass.analysis.leakage import run_leakage_detection

report = run_leakage_detection(X_train, y_train, X_test, y_test, max_samples=2000)
```

---

## `leakage_report.json` Schema

The saved file is a JSON object with the following structure:

```json
{
  "passed": false,
  "num_passed": 7,
  "num_failed": 1,
  "total_duration_ms": 145.3,
  "results": [
    {
      "check_id": "EXACT_OVERLAP",
      "title": "Train/test exact-sample overlap",
      "passed": true,
      "severity": "CRITICAL",
      "details": {
        "num_overlapping": 0,
        "train_size": 2000,
        "test_size": 500
      },
      "recommendation": "",
      "duration_ms": 12.4
    },
    {
      "check_id": "TARGET_CORRELATION",
      "title": "Feature-target correlation check",
      "passed": false,
      "severity": "HIGH",
      "details": {
        "num_suspicious": 1,
        "threshold": 0.95,
        "suspicious_features": [{"feature_idx": 3, "correlation": 0.971}],
        "top_correlations": []
      },
      "recommendation": "1 features have suspiciously high correlation with targets (|r| > 0.95).",
      "duration_ms": 8.1
    }
  ]
}
```

Each entry in `results` corresponds to one of the 8 checks, in the order they are run.
