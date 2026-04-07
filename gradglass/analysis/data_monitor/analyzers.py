from __future__ import annotations

import itertools
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from gradglass.analysis.data_monitor.fingerprinting import cosine_similarity
from gradglass.analysis.data_monitor.inspectors import SampleObservation
from gradglass.analysis.data_monitor.models import (
    CheckCategory,
    CheckSeverity,
    CheckStatus,
    CompositionSlice,
    DatasetMonitorConfig,
    MonitorCheckResult,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGE_ORDER,
    PipelineStage,
    PipelineStageSnapshot,
    RecommendationItem,
    SplitComparisonSummary,
    TaskType,
)


@dataclass
class StageRecord:
    stage: PipelineStage
    split: str
    snapshot: PipelineStageSnapshot
    observations: list[SampleObservation]
    metadata: dict[str, Any]
    adapter_errors: list[str]


def _histogram(values: list[float], bins: int) -> dict[str, Any]:
    if not values:
        return {"available": False, "bins": [], "counts": []}
    counts, edges = np.histogram(np.asarray(values, dtype=np.float32), bins=min(bins, max(1, len(set(values)))))
    labels = [f"{round(float(edges[idx]), 4)}..{round(float(edges[idx + 1]), 4)}" for idx in range(len(edges) - 1)]
    return {
        "available": True,
        "bins": labels,
        "counts": [int(count) for count in counts.tolist()],
        "min": round(float(np.min(values)), 6),
        "max": round(float(np.max(values)), 6),
        "mean": round(float(np.mean(values)), 6),
    }


def _status_from_issue(failed: bool, warn: bool = False) -> CheckStatus:
    if failed:
        return CheckStatus.FAILED
    if warn:
        return CheckStatus.WARNING
    return CheckStatus.PASSED


def _safe_round(value: Any, digits: int = 6):
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except Exception:
        return value


def _flatten_labels(observations: list[SampleObservation]) -> list[Any]:
    labels: list[Any] = []
    for obs in observations:
        label = obs.normalized_label
        if label is None:
            continue
        if isinstance(label, list):
            if label and all(not isinstance(item, (list, dict)) for item in label):
                labels.extend(label)
            else:
                labels.append(tuple(label))
        else:
            labels.append(label)
    return labels


def _labels_are_numeric(labels: list[Any]) -> bool:
    return bool(labels) and all(isinstance(item, (int, float, np.integer, np.floating)) for item in labels)


def _build_class_distribution(labels: list[Any]) -> dict[str, float]:
    if not labels:
        return {}
    counts = Counter(str(item) for item in labels)
    total = sum(counts.values())
    return {key: round(count / max(total, 1), 4) for key, count in counts.items()}


def _collect_numeric_matrix(observations: list[SampleObservation]) -> tuple[list[str], np.ndarray]:
    feature_index: dict[str, int] = {}
    row_maps: list[dict[str, float]] = []
    for obs in observations:
        if obs.feature_vector is None or obs.feature_vector.size == 0:
            continue
        if obs.feature_names and len(obs.feature_names) == obs.feature_vector.size:
            mapping = {name: float(obs.feature_vector[idx]) for idx, name in enumerate(obs.feature_names)}
        else:
            mapping = {f"feature_{idx}": float(value) for idx, value in enumerate(obs.feature_vector.tolist())}
        row_maps.append(mapping)
        for name in mapping:
            if name not in feature_index:
                feature_index[name] = len(feature_index)
    if not row_maps:
        return [], np.zeros((0, 0), dtype=np.float32)
    matrix = np.full((len(row_maps), len(feature_index)), np.nan, dtype=np.float32)
    for row_idx, mapping in enumerate(row_maps):
        for name, value in mapping.items():
            matrix[row_idx, feature_index[name]] = value
    names = [name for name, _ in sorted(feature_index.items(), key=lambda item: item[1])]
    return names, matrix


def _collect_schema_summary(observations: list[SampleObservation]) -> dict[str, Any]:
    field_counter = Counter()
    for obs in observations:
        field_counter.update(obs.schema_fields)
    return {"fields": sorted(field_counter.keys()), "field_counts": dict(field_counter)}


def _compute_snapshot_status(
    snapshot: PipelineStageSnapshot, observations: list[SampleObservation], adapter_errors: list[str]
) -> CheckStatus:
    if adapter_errors:
        return CheckStatus.WARNING if observations else CheckStatus.UNKNOWN
    if snapshot.sample_count == 0 or (snapshot.observed_sample_count == 0 and snapshot.sample_count not in (0, None)):
        return CheckStatus.UNKNOWN
    if snapshot.null_missing_rate is not None and snapshot.null_missing_rate > 0.25:
        return CheckStatus.WARNING
    return CheckStatus.PASSED


def enrich_stage_snapshots(records: list[StageRecord]) -> list[StageRecord]:
    previous_by_split: dict[str, StageRecord] = {}
    sorted_records = sorted(records, key=lambda item: (PIPELINE_STAGE_ORDER.index(item.stage), item.split))
    for record in sorted_records:
        previous = previous_by_split.get(record.split)
        record.snapshot.stage_label = PIPELINE_STAGE_LABELS[record.stage]
        record.snapshot.stage_index = PIPELINE_STAGE_ORDER.index(record.stage)
        total_count = record.snapshot.sample_count
        observed = record.snapshot.observed_sample_count
        if total_count is None:
            record.snapshot.sample_coverage = 1.0 if observed else 0.0
        else:
            record.snapshot.sample_coverage = round(float(observed / max(total_count, 1)), 4)
        if previous is not None:
            record.snapshot.parent_stage = previous.stage
            prev_fields = set(_collect_schema_summary(previous.observations)["fields"])
            curr_fields = set(_collect_schema_summary(record.observations)["fields"])
            added_fields = sorted(curr_fields - prev_fields)
            removed_fields = sorted(prev_fields - curr_fields)
            if added_fields:
                record.snapshot.schema_changes.append({"type": "added_fields", "fields": added_fields})
            if removed_fields:
                record.snapshot.schema_changes.append({"type": "removed_fields", "fields": removed_fields})
            full_prev = previous.snapshot.sample_coverage == 1.0
            full_curr = record.snapshot.sample_coverage == 1.0
            if full_prev and full_curr:
                prev_count = previous.snapshot.sample_count or len(previous.observations)
                curr_count = record.snapshot.sample_count or len(record.observations)
                record.snapshot.dropped_samples = max(0, prev_count - curr_count)
                record.snapshot.added_samples = max(0, curr_count - prev_count)
                prev_fps = {obs.normalized_fingerprint for obs in previous.observations if obs.normalized_fingerprint}
                curr_fps = {obs.normalized_fingerprint for obs in record.observations if obs.normalized_fingerprint}
                record.snapshot.metrics["observed_exact_overlap"] = len(prev_fps & curr_fps)
        previous_by_split[record.split] = record
        record.snapshot.status = _compute_snapshot_status(record.snapshot, record.observations, record.adapter_errors)
    return sorted_records


def build_composition_slice(
    stage: PipelineStage,
    split: str,
    observations: list[SampleObservation],
    config: DatasetMonitorConfig,
    metadata: Optional[dict[str, Any]] = None,
) -> CompositionSlice:
    metadata = metadata or {}
    modality_counts = Counter(obs.modality.value for obs in observations)
    total = max(sum(modality_counts.values()), 1)
    modality_proportions = {key: round(value / total, 4) for key, value in modality_counts.items()}

    labels = _flatten_labels(observations)
    class_distribution: dict[str, float] = {}
    regression_target_distribution: dict[str, Any] = {}
    if labels:
        if _labels_are_numeric(labels) and len(set(labels)) > min(20, max(5, len(labels) // 10)):
            regression_target_distribution = _histogram([float(item) for item in labels], config.histogram_bins)
        else:
            class_distribution = _build_class_distribution(labels)

    seq_lengths = [int(obs.metrics["sequence_length"]) for obs in observations if "sequence_length" in obs.metrics]
    image_heights = [int(obs.metrics["image_height"]) for obs in observations if "image_height" in obs.metrics]
    image_widths = [int(obs.metrics["image_width"]) for obs in observations if "image_width" in obs.metrics]
    image_aspects = [
        float(obs.metrics["aspect_ratio"]) for obs in observations if obs.metrics.get("aspect_ratio") is not None
    ]
    audio_durations = [float(obs.metrics["duration_s"]) for obs in observations if "duration_s" in obs.metrics]
    audio_sample_rates = Counter(
        str(obs.metrics.get("sample_rate")) for obs in observations if obs.metrics.get("sample_rate")
    )
    missing_rates = [obs.missing_rate for obs in observations]

    field_missing_counter = Counter()
    categorical_values: dict[str, set[str]] = defaultdict(set)
    numeric_values: dict[str, list[float]] = defaultdict(list)
    provenance_counts = Counter()

    for obs in observations:
        if isinstance(obs.sample, dict):
            for key, value in obs.sample.items():
                if value is None:
                    field_missing_counter[key] += 1
                elif isinstance(value, (str, bytes)):
                    categorical_values[key].add(str(value))
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    numeric_values[key].append(float(value))
            provenance = metadata.get("provenance") or metadata.get("source")
            if provenance:
                provenance_counts[str(provenance)] += 1
        if obs.feature_names and obs.feature_vector is not None:
            for idx, name in enumerate(obs.feature_names):
                if idx < obs.feature_vector.size:
                    numeric_values[name].append(float(obs.feature_vector[idx]))

    numerical_feature_stats = {}
    outlier_features = []
    for name, values in numeric_values.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            continue
        stat = {
            "mean": _safe_round(np.nanmean(arr)),
            "std": _safe_round(np.nanstd(arr)),
            "min": _safe_round(np.nanmin(arr)),
            "max": _safe_round(np.nanmax(arr)),
            "p10": _safe_round(np.nanpercentile(arr, 10)),
            "p50": _safe_round(np.nanpercentile(arr, 50)),
            "p90": _safe_round(np.nanpercentile(arr, 90)),
        }
        numerical_feature_stats[name] = stat
        spread = abs(float(stat["p90"] or 0.0) - float(stat["p10"] or 0.0))
        std_value = float(stat["std"] or 0.0)
        if spread <= 1e-6 and abs(std_value) <= 1e-6:
            continue
        outlier_features.append({"feature": name, "spread": round(spread, 6), "std": stat["std"]})
    outlier_features.sort(key=lambda item: float(item.get("spread", 0.0)), reverse=True)

    source_sample_count = metadata.get("source_sample_count")
    try:
        source_sample_count = int(source_sample_count) if source_sample_count is not None else None
    except (TypeError, ValueError):
        source_sample_count = None
    composition_sample_count = len(observations)
    if source_sample_count is not None:
        composition_sample_count = max(composition_sample_count, source_sample_count)

    return CompositionSlice(
        stage=stage,
        split=split,
        sample_count=composition_sample_count,
        modality_proportions=modality_proportions,
        class_distribution=class_distribution,
        regression_target_distribution=regression_target_distribution,
        sequence_length_distribution=_histogram([float(length) for length in seq_lengths], config.histogram_bins),
        image_size_distribution={
            "available": bool(image_heights and image_widths),
            "heights": _histogram([float(item) for item in image_heights], config.histogram_bins),
            "widths": _histogram([float(item) for item in image_widths], config.histogram_bins),
        },
        image_aspect_ratio_distribution=_histogram(image_aspects, config.histogram_bins),
        audio_duration_distribution=_histogram(audio_durations, config.histogram_bins),
        audio_sample_rates={key: int(value) for key, value in audio_sample_rates.items()},
        missingness_patterns={
            "average_missing_rate": _safe_round(float(np.mean(missing_rates)) if missing_rates else 0.0),
            "top_missing_fields": field_missing_counter.most_common(config.top_feature_limit),
        },
        categorical_cardinality={key: len(value) for key, value in sorted(categorical_values.items())},
        numerical_feature_stats=numerical_feature_stats,
        provenance={
            key: round(value / max(sum(provenance_counts.values()), 1), 4) for key, value in provenance_counts.items()
        },
        outlier_features=outlier_features[: config.top_feature_limit],
    )


def build_split_comparisons(
    latest_by_split: dict[str, StageRecord], config: DatasetMonitorConfig
) -> list[SplitComparisonSummary]:
    split_names = list(latest_by_split)
    if len(split_names) < 2:
        return []
    if "train" in latest_by_split:
        pairs = [("train", split_name) for split_name in split_names if split_name != "train"]
    else:
        pairs = list(itertools.combinations(split_names, 2))
    results = []
    for split_a, split_b in pairs:
        record_a = latest_by_split[split_a]
        record_b = latest_by_split[split_b]
        labels_a = _flatten_labels(record_a.observations)
        labels_b = _flatten_labels(record_b.observations)
        dist_a = _build_class_distribution(labels_a) if labels_a and not _labels_are_numeric(labels_a) else {}
        dist_b = _build_class_distribution(labels_b) if labels_b and not _labels_are_numeric(labels_b) else {}
        all_labels = sorted(set(dist_a) | set(dist_b))
        label_diff = {label: round(abs(dist_a.get(label, 0.0) - dist_b.get(label, 0.0)), 4) for label in all_labels}

        feature_names_a, matrix_a = _collect_numeric_matrix(record_a.observations)
        feature_names_b, matrix_b = _collect_numeric_matrix(record_b.observations)
        numeric_drift = {"available": False, "top_drifted_features": []}
        if feature_names_a and feature_names_b:
            common_features = [name for name in feature_names_a if name in set(feature_names_b)]
            if common_features:
                idx_a = {name: feature_names_a.index(name) for name in common_features}
                idx_b = {name: feature_names_b.index(name) for name in common_features}
                drift_scores = []
                for name in common_features:
                    col_a = matrix_a[:, idx_a[name]]
                    col_b = matrix_b[:, idx_b[name]]
                    mean_diff = abs(float(np.nanmean(col_a) - np.nanmean(col_b)))
                    std_a = float(np.nanstd(col_a))
                    std_b = float(np.nanstd(col_b))
                    if std_a <= 1e-8 and std_b <= 1e-8:
                        std_ratio = 1.0
                    else:
                        std_ratio = float(max(std_a, std_b) / (min(std_a, std_b) + 1e-8))
                    drift_item = {"feature": name, "mean_diff": round(mean_diff, 6), "std_ratio": round(std_ratio, 4)}
                    if drift_item["mean_diff"] > 1e-6 or abs(drift_item["std_ratio"] - 1.0) > 0.05:
                        drift_scores.append(drift_item)
                drift_scores.sort(key=lambda item: (item["mean_diff"], abs(item["std_ratio"] - 1.0)), reverse=True)
                numeric_drift = {
                    "available": bool(drift_scores),
                    "top_drifted_features": drift_scores[: config.top_feature_limit],
                }

        modality_a = record_a.snapshot.modality_metadata.get("modality_proportions", {})
        modality_b = record_b.snapshot.modality_metadata.get("modality_proportions", {})
        modality_keys = sorted(set(modality_a) | set(modality_b))
        modality_diff = {
            key: round(abs(modality_a.get(key, 0.0) - modality_b.get(key, 0.0)), 4) for key in modality_keys
        }
        summary_parts = []
        if label_diff:
            summary_parts.append(f"max label diff {_safe_round(max(label_diff.values()), 4)}")
        if numeric_drift.get("top_drifted_features"):
            top_name = numeric_drift["top_drifted_features"][0]["feature"]
            summary_parts.append(f"top drift feature {top_name}")
        results.append(
            SplitComparisonSummary(
                split_a=split_a,
                split_b=split_b,
                label_distribution_diff={
                    "split_a_distribution": dist_a,
                    "split_b_distribution": dist_b,
                    "per_label_abs_diff": label_diff,
                    "max_abs_diff": max(label_diff.values()) if label_diff else None,
                },
                numeric_drift_summary=numeric_drift,
                modality_diff=modality_diff,
                summary=", ".join(summary_parts) if summary_parts else f"Compared {split_a} vs {split_b}",
            )
        )
    return results


def _latest_by_split(records: list[StageRecord]) -> dict[str, StageRecord]:
    latest: dict[str, StageRecord] = {}
    for record in sorted(records, key=lambda item: PIPELINE_STAGE_ORDER.index(item.stage)):
        latest[record.split] = record
    return latest


def _make_check(
    *,
    name: str,
    category: CheckCategory,
    severity: CheckSeverity,
    status: CheckStatus,
    summary: str,
    payload: Optional[dict[str, Any]] = None,
    recommendation: str | None = None,
    evidence: Optional[list[dict[str, Any]]] = None,
    metrics: Optional[dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> MonitorCheckResult:
    return MonitorCheckResult(
        name=name,
        category=category,
        severity=severity,
        status=status,
        summary=summary,
        payload=payload or {},
        recommendation=recommendation,
        evidence=evidence or [],
        metrics=metrics or {},
        duration_ms=round(float(duration_ms), 2),
    )


def _unknown_check(name: str, severity: CheckSeverity, reason: str) -> MonitorCheckResult:
    return _make_check(
        name=name,
        category=CheckCategory.LEAKAGE,
        severity=severity,
        status=CheckStatus.UNKNOWN,
        summary=reason,
        recommendation="Record both train and test stage snapshots with samples and labels to enable this check.",
    )


def _duplicate_groups(observations: list[SampleObservation], *, normalized: bool = False) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for obs in observations:
        key = obs.normalized_fingerprint if normalized else obs.exact_fingerprint
        if key:
            groups[key].append(obs.index)
    return {key: indices for key, indices in groups.items() if len(indices) > 1}


def _group_labels_for_distribution(
    observations: list[SampleObservation], bins: int
) -> tuple[dict[str, float], dict[str, Any]]:
    labels = _flatten_labels(observations)
    if not labels:
        return {}, {}
    if _labels_are_numeric(labels) and len(set(labels)) > min(20, max(5, len(labels) // 10)):
        hist = _histogram([float(item) for item in labels], bins)
        counts = hist.get("counts", [])
        total = sum(counts)
        distribution = {hist["bins"][idx]: round(count / max(total, 1), 4) for idx, count in enumerate(counts)}
        return distribution, hist
    return _build_class_distribution(labels), {}


def _numeric_labels(observations: list[SampleObservation]) -> np.ndarray:
    labels = []
    encoded_labels = {}
    next_index = 0
    for obs in observations:
        label = obs.normalized_label
        if label is None:
            continue
        if isinstance(label, list):
            label = tuple(label)
        if isinstance(label, (int, float, np.integer, np.floating)):
            labels.append(float(label))
        else:
            if label not in encoded_labels:
                encoded_labels[label] = next_index
                next_index += 1
            labels.append(float(encoded_labels[label]))
    return np.asarray(labels, dtype=np.float32)


def run_leakage_checks(
    records: list[StageRecord], task: TaskType, config: DatasetMonitorConfig
) -> list[MonitorCheckResult]:
    latest = _latest_by_split(records)
    train = latest.get("train")
    test = latest.get("test")
    if train is None or test is None:
        return [
            _unknown_check(
                "Train/test exact-sample overlap", CheckSeverity.CRITICAL, "Train or test split is unavailable."
            ),
            _unknown_check(
                "Duplicate samples within training set", CheckSeverity.MEDIUM, "Training split is unavailable."
            ),
            _unknown_check("Duplicate samples within test set", CheckSeverity.MEDIUM, "Test split is unavailable."),
            _unknown_check(
                "Train/test near-duplicate samples", CheckSeverity.HIGH, "Train or test split is unavailable."
            ),
            _unknown_check(
                "Train/test label-distribution consistency",
                CheckSeverity.MEDIUM,
                "Train or test labels are unavailable.",
            ),
            _unknown_check(
                "Train/test feature-statistics consistency",
                CheckSeverity.MEDIUM,
                "Comparable numeric features are unavailable.",
            ),
            _unknown_check(
                "Feature-target correlation check", CheckSeverity.HIGH, "Labels or numeric features are unavailable."
            ),
            _unknown_check(
                "Preprocessing / scaler leakage", CheckSeverity.HIGH, "Comparable numeric features are unavailable."
            ),
        ]

    checks: list[MonitorCheckResult] = []

    start = time.time()
    train_exact = {obs.exact_fingerprint: obs.index for obs in train.observations if obs.exact_fingerprint}
    overlapping = [
        {"test_index": obs.index, "train_index": train_exact[obs.exact_fingerprint]}
        for obs in test.observations
        if obs.exact_fingerprint in train_exact
    ]
    checks.append(
        _make_check(
            name="Train/test exact-sample overlap",
            category=CheckCategory.LEAKAGE,
            severity=CheckSeverity.CRITICAL,
            status=_status_from_issue(bool(overlapping)),
            summary="No exact overlaps detected between train and test."
            if not overlapping
            else f"Detected {len(overlapping)} exact overlaps between train and test.",
            payload={
                "num_overlapping": len(overlapping),
                "train_size": len(train.observations),
                "test_size": len(test.observations),
            },
            recommendation="Remove duplicated examples from the evaluation split and regenerate metrics."
            if overlapping
            else None,
            evidence=overlapping[: config.evidence_limit],
            metrics={"overlap_rate": _safe_round(len(overlapping) / max(len(test.observations), 1), 6)},
            duration_ms=(time.time() - start) * 1000,
        )
    )

    start = time.time()
    train_dups = _duplicate_groups(train.observations)
    train_extra = sum(len(indices) - 1 for indices in train_dups.values())
    checks.append(
        _make_check(
            name="Duplicate samples within training set",
            category=CheckCategory.LEAKAGE,
            severity=CheckSeverity.MEDIUM,
            status=_status_from_issue(False, warn=bool(train_dups)),
            summary="Training split has no exact duplicate groups."
            if not train_dups
            else f"Training split contains {len(train_dups)} duplicate groups and {train_extra} extra copies.",
            payload={
                "num_duplicate_groups": len(train_dups),
                "total_extra_copies": train_extra,
                "train_size": len(train.observations),
            },
            recommendation="Deduplicate training samples or down-weight repeated records to reduce training bias."
            if train_dups
            else None,
            evidence=[{"indices": indices[:10]} for indices in list(train_dups.values())[: config.evidence_limit]],
            duration_ms=(time.time() - start) * 1000,
        )
    )

    start = time.time()
    test_dups = _duplicate_groups(test.observations)
    test_extra = sum(len(indices) - 1 for indices in test_dups.values())
    checks.append(
        _make_check(
            name="Duplicate samples within test set",
            category=CheckCategory.LEAKAGE,
            severity=CheckSeverity.MEDIUM,
            status=_status_from_issue(False, warn=bool(test_dups)),
            summary="Test split has no exact duplicate groups."
            if not test_dups
            else f"Test split contains {len(test_dups)} duplicate groups and {test_extra} extra copies.",
            payload={
                "num_duplicate_groups": len(test_dups),
                "total_extra_copies": test_extra,
                "test_size": len(test.observations),
            },
            recommendation="Deduplicate evaluation samples to avoid overweighting repeated examples."
            if test_dups
            else None,
            evidence=[{"indices": indices[:10]} for indices in list(test_dups.values())[: config.evidence_limit]],
            duration_ms=(time.time() - start) * 1000,
        )
    )

    start = time.time()
    train_blocks: dict[tuple[str, int], list[SampleObservation]] = defaultdict(list)
    for obs in train.observations:
        modality = obs.modality.value
        signature = np.asarray(obs.approximate_signature, dtype=np.float32)
        if signature.size == 0:
            continue
        bucket = int(np.round(float(signature[0]) * 100))
        train_blocks[(modality, bucket)].append(obs)

    near_pairs = []
    candidates_seen = 0
    for obs in test.observations:
        modality = obs.modality.value
        signature = np.asarray(obs.approximate_signature, dtype=np.float32)
        if signature.size == 0:
            continue
        bucket = int(np.round(float(signature[0]) * 100))
        candidates = train_blocks.get((modality, bucket), [])
        for candidate in candidates:
            if candidates_seen >= config.pairwise_candidate_budget:
                break
            candidates_seen += 1
            similarity = cosine_similarity(candidate.approximate_signature, obs.approximate_signature)
            if (
                similarity >= config.near_duplicate_similarity_threshold
                and candidate.exact_fingerprint != obs.exact_fingerprint
            ):
                near_pairs.append(
                    {
                        "train_index": candidate.index,
                        "test_index": obs.index,
                        "similarity": round(similarity, 4),
                        "modality": modality,
                    }
                )
                if len(near_pairs) >= config.evidence_limit:
                    break
        if candidates_seen >= config.pairwise_candidate_budget or len(near_pairs) >= config.evidence_limit:
            break

    checks.append(
        _make_check(
            name="Train/test near-duplicate samples",
            category=CheckCategory.LEAKAGE,
            severity=CheckSeverity.HIGH,
            status=_status_from_issue(False, warn=bool(near_pairs)),
            summary="No near-duplicate train/test pairs were detected."
            if not near_pairs
            else f"Detected {len(near_pairs)} near-duplicate train/test pairs within the candidate budget.",
            payload={
                "num_near_duplicates": len(near_pairs),
                "candidate_pairs_examined": candidates_seen,
                "similarity_threshold": config.near_duplicate_similarity_threshold,
            },
            recommendation="Inspect these highly similar examples and tighten deduplication or split logic before evaluation."
            if near_pairs
            else None,
            evidence=near_pairs,
            metrics={"candidate_budget": config.pairwise_candidate_budget},
            duration_ms=(time.time() - start) * 1000,
        )
    )

    start = time.time()
    train_dist, train_reg_hist = _group_labels_for_distribution(train.observations, config.histogram_bins)
    test_dist, test_reg_hist = _group_labels_for_distribution(test.observations, config.histogram_bins)
    if not train_dist and not test_dist and not train_reg_hist and not test_reg_hist:
        checks.append(
            _unknown_check(
                "Train/test label-distribution consistency",
                CheckSeverity.MEDIUM,
                "Train or test labels are unavailable.",
            )
        )
    else:
        labels = sorted(set(train_dist) | set(test_dist))
        diffs = {label: round(abs(train_dist.get(label, 0.0) - test_dist.get(label, 0.0)), 4) for label in labels}
        max_diff = max(diffs.values()) if diffs else 0.0
        warn = max_diff >= 0.15
        checks.append(
            _make_check(
                name="Train/test label-distribution consistency",
                category=CheckCategory.LEAKAGE,
                severity=CheckSeverity.MEDIUM,
                status=_status_from_issue(False, warn=warn),
                summary="Train and test label distributions look consistent."
                if not warn
                else f"Train and test label distributions diverge by up to {max_diff:.2%}.",
                payload={
                    "train_distribution": train_dist,
                    "test_distribution": test_dist,
                    "train_regression_buckets": train_reg_hist,
                    "test_regression_buckets": test_reg_hist,
                    "max_absolute_diff": round(max_diff, 4),
                },
                recommendation="Revisit split strategy or stratification if evaluation is meant to mirror training distribution."
                if warn
                else None,
                evidence=[
                    {"label": label, "absolute_diff": diff}
                    for label, diff in sorted(diffs.items(), key=lambda item: item[1], reverse=True)[
                        : config.top_feature_limit
                    ]
                ],
                duration_ms=(time.time() - start) * 1000,
            )
        )

    start = time.time()
    train_feature_names, train_matrix = _collect_numeric_matrix(train.observations)
    test_feature_names, test_matrix = _collect_numeric_matrix(test.observations)
    common_feature_names = [name for name in train_feature_names if name in set(test_feature_names)]
    if not common_feature_names:
        checks.append(
            _unknown_check(
                "Train/test feature-statistics consistency",
                CheckSeverity.MEDIUM,
                "Comparable numeric features are unavailable.",
            )
        )
    else:
        idx_train = {name: train_feature_names.index(name) for name in common_feature_names}
        idx_test = {name: test_feature_names.index(name) for name in common_feature_names}
        feature_drifts = []
        mean_diffs = []
        std_ratios = []
        for name in common_feature_names:
            train_col = train_matrix[:, idx_train[name]]
            test_col = test_matrix[:, idx_test[name]]
            train_mean = float(np.nanmean(train_col))
            test_mean = float(np.nanmean(test_col))
            train_std = float(np.nanstd(train_col))
            test_std = float(np.nanstd(test_col))
            mean_diff = abs(train_mean - test_mean)
            std_ratio = max(train_std, test_std) / (min(train_std, test_std) + 1e-8)
            mean_diffs.append(mean_diff)
            std_ratios.append(std_ratio)
            feature_drifts.append(
                {
                    "feature": name,
                    "train_mean": round(train_mean, 6),
                    "test_mean": round(test_mean, 6),
                    "mean_diff": round(mean_diff, 6),
                    "std_ratio": round(std_ratio, 4),
                }
            )
        feature_drifts.sort(key=lambda item: (item["mean_diff"], item["std_ratio"]), reverse=True)
        mean_diff_avg = float(np.mean(mean_diffs)) if mean_diffs else 0.0
        std_ratio_avg = float(np.mean(std_ratios)) if std_ratios else 1.0
        warn = mean_diff_avg >= 0.5 or std_ratio_avg >= 2.0
        checks.append(
            _make_check(
                name="Train/test feature-statistics consistency",
                category=CheckCategory.LEAKAGE,
                severity=CheckSeverity.MEDIUM,
                status=_status_from_issue(False, warn=warn),
                summary="Train and test feature statistics are aligned."
                if not warn
                else f"Feature stats drift detected: mean_diff_avg={mean_diff_avg:.4f}, std_ratio_avg={std_ratio_avg:.4f}.",
                payload={
                    "mean_diff_avg": round(mean_diff_avg, 6),
                    "std_ratio_avg": round(std_ratio_avg, 4),
                    "num_common_features": len(common_feature_names),
                },
                recommendation="Validate preprocessing parity and investigate the most drifted features."
                if warn
                else None,
                evidence=feature_drifts[: config.top_feature_limit],
                duration_ms=(time.time() - start) * 1000,
            )
        )

    start = time.time()
    combined_observations = train.observations + test.observations
    feature_names, feature_matrix = _collect_numeric_matrix(combined_observations)
    numeric_labels = _numeric_labels(combined_observations)
    if not feature_names or feature_matrix.shape[0] == 0 or numeric_labels.size == 0:
        checks.append(
            _unknown_check(
                "Feature-target correlation check", CheckSeverity.HIGH, "Labels or numeric features are unavailable."
            )
        )
    else:
        usable_rows = min(feature_matrix.shape[0], numeric_labels.size)
        feature_matrix = feature_matrix[:usable_rows]
        numeric_labels = numeric_labels[:usable_rows]
        centered_labels = numeric_labels - float(np.mean(numeric_labels))
        label_std = float(np.std(centered_labels)) + 1e-8
        suspicious = []
        top_corrs = []
        for idx, name in enumerate(feature_names):
            column = feature_matrix[:, idx]
            if np.all(np.isnan(column)):
                continue
            col = np.nan_to_num(column, nan=float(np.nanmean(column)))
            centered = col - float(np.mean(col))
            corr = float(np.mean(centered * centered_labels) / ((float(np.std(col)) + 1e-8) * label_std))
            entry = {"feature": name, "correlation": round(corr, 6)}
            top_corrs.append(entry)
            if abs(corr) >= config.leakage_correlation_threshold:
                suspicious.append(entry)

        if isinstance(combined_observations[0].sample, dict):
            for name in feature_names[:]:
                values = []
                for obs in combined_observations[:usable_rows]:
                    if isinstance(obs.sample, dict):
                        values.append(obs.sample.get(name))
                labels_normalized = [obs.normalized_label for obs in combined_observations[:usable_rows]]
                matches = 0
                comparable = 0
                for value, label in zip(values, labels_normalized):
                    if value is None or label is None:
                        continue
                    comparable += 1
                    if str(value).strip().lower() == str(label).strip().lower():
                        matches += 1
                if comparable and matches / comparable >= 0.95:
                    suspicious.append({"feature": name, "correlation": 1.0, "kind": "exact_target_copy"})
        top_corrs.sort(key=lambda item: abs(item["correlation"]), reverse=True)
        suspicious.sort(key=lambda item: abs(item.get("correlation", 0.0)), reverse=True)
        warn = bool(suspicious)
        checks.append(
            _make_check(
                name="Feature-target correlation check",
                category=CheckCategory.LEAKAGE,
                severity=CheckSeverity.HIGH,
                status=_status_from_issue(bool(suspicious)),
                summary="No suspicious feature-target correlations were detected."
                if not warn
                else f"Detected {len(suspicious)} suspicious feature-target correlations or proxy features.",
                payload={"num_suspicious": len(suspicious), "threshold": config.leakage_correlation_threshold},
                recommendation="Inspect the highlighted features for target copies, future information, or label-derived proxies."
                if warn
                else None,
                evidence=suspicious[: config.evidence_limit],
                metrics={"top_correlations": top_corrs[: config.top_feature_limit]},
                duration_ms=(time.time() - start) * 1000,
            )
        )

    start = time.time()
    if not common_feature_names:
        checks.append(
            _unknown_check(
                "Preprocessing / scaler leakage", CheckSeverity.HIGH, "Comparable numeric features are unavailable."
            )
        )
    else:
        idx_train = {name: train_feature_names.index(name) for name in common_feature_names}
        idx_test = {name: test_feature_names.index(name) for name in common_feature_names}
        train_subset = np.column_stack([train_matrix[:, idx_train[name]] for name in common_feature_names])
        test_subset = np.column_stack([test_matrix[:, idx_test[name]] for name in common_feature_names])
        train_mean = np.nanmean(train_subset, axis=0)
        test_mean = np.nanmean(test_subset, axis=0)
        train_std = np.nanstd(train_subset, axis=0)
        avg_train_mean_abs = float(np.nanmean(np.abs(train_mean)))
        avg_train_std_diff = float(np.nanmean(np.abs(train_std - 1.0)))
        avg_test_mean_abs = float(np.nanmean(np.abs(test_mean)))
        combined_mean = (len(train_subset) * train_mean + len(test_subset) * test_mean) / max(
            len(train_subset) + len(test_subset), 1
        )
        avg_combined_abs = float(np.nanmean(np.abs(combined_mean)))
        train_looks_standardized = avg_train_mean_abs < 0.15 and avg_train_std_diff < 0.15
        leakage = bool(
            train_looks_standardized
            and avg_train_mean_abs >= 1e-4
            and avg_test_mean_abs < 0.15
            and avg_combined_abs < 0.05
        )
        status = CheckStatus.FAILED if leakage else CheckStatus.PASSED
        if not train_looks_standardized:
            status = CheckStatus.PASSED
        summary = "No evidence of scaler leakage detected."
        recommendation = None
        if leakage:
            summary = "Train and test splits appear to have been standardized using combined statistics."
            recommendation = (
                "Fit normalization and scaling only on the training split, then transform validation/test separately."
            )
        elif not train_looks_standardized:
            summary = "Standardization heuristics were not applicable for this stage."
        checks.append(
            _make_check(
                name="Preprocessing / scaler leakage",
                category=CheckCategory.LEAKAGE,
                severity=CheckSeverity.HIGH,
                status=status,
                summary=summary,
                payload={
                    "avg_train_mean_abs": round(avg_train_mean_abs, 6),
                    "avg_train_std_diff": round(avg_train_std_diff, 6),
                    "avg_test_mean_abs": round(avg_test_mean_abs, 6),
                    "avg_combined_mean_abs": round(avg_combined_abs, 6),
                },
                recommendation=recommendation,
                duration_ms=(time.time() - start) * 1000,
            )
        )

    return checks


def build_recommendations(checks: list[MonitorCheckResult], config: DatasetMonitorConfig) -> list[RecommendationItem]:
    severity_weight = {
        CheckSeverity.CRITICAL: 5.0,
        CheckSeverity.HIGH: 4.0,
        CheckSeverity.MEDIUM: 3.0,
        CheckSeverity.LOW: 2.0,
        CheckSeverity.INFO: 1.0,
    }
    recommendations: list[RecommendationItem] = []
    for check in checks:
        if check.status not in (CheckStatus.FAILED, CheckStatus.WARNING):
            continue
        confidence = 0.7 if check.status == CheckStatus.FAILED else 0.55
        affected_splits = []
        for key in ("split", "split_a", "split_b"):
            value = check.payload.get(key)
            if isinstance(value, str):
                affected_splits.append(value)
        evidence_volume = len(check.evidence)
        sample_volume = 0
        for key in ("train_size", "test_size", "sample_count"):
            value = check.payload.get(key)
            if isinstance(value, int):
                sample_volume += value
        score = (
            severity_weight[check.severity] * 10.0
            + confidence * 5.0
            + min(evidence_volume, config.evidence_limit) * 0.05
            + min(sample_volume, 5000) / 5000.0
        )
        recommendations.append(
            RecommendationItem(
                title=check.name,
                severity=check.severity,
                summary=check.recommendation or check.summary,
                confidence=confidence,
                affected_splits=sorted(set(affected_splits)),
                related_checks=[check.name],
                next_steps=[check.recommendation] if check.recommendation else [],
                score=round(score, 4),
            )
        )
    recommendations.sort(key=lambda item: (item.score, item.confidence), reverse=True)
    return recommendations
