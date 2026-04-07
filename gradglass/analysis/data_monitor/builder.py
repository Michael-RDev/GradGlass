from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from gradglass.analysis.data_monitor.adapters import adapt_input
from gradglass.analysis.data_monitor.analyzers import (
    StageRecord,
    build_composition_slice,
    build_recommendations,
    build_split_comparisons,
    enrich_stage_snapshots,
    run_leakage_checks,
)
from gradglass.analysis.data_monitor.inspectors import detect_modality, inspect_sample
from gradglass.analysis.data_monitor.models import (
    CheckCategory,
    CheckStatus,
    CompositionSlice,
    CompositionSummary,
    DashboardPanel,
    DashboardViewModel,
    DatasetMonitorConfig,
    DatasetMonitorMetadata,
    DatasetMonitorReport,
    PIPELINE_STAGE_LABELS,
    PIPELINE_STAGE_ORDER,
    PipelineStage,
    PipelineStageSnapshot,
    TaskType,
)


@dataclass
class StageCapture:
    stage: PipelineStage
    split: str
    data: Any
    labels: Any
    metadata: dict[str, Any]
    latency_ms: Optional[float]
    parent_stage: Optional[PipelineStage]
    sample_limit: Optional[int]


def _normalize_stage(stage: PipelineStage | str) -> PipelineStage:
    if isinstance(stage, PipelineStage):
        return stage
    return PipelineStage(stage)


class DatasetMonitorBuilder:
    def __init__(
        self,
        task: TaskType | str,
        *,
        dataset_name: str | None = None,
        task_hint: str | None = None,
        config: DatasetMonitorConfig | dict[str, Any] | None = None,
        run_dir: Path | None = None,
        run_id: str | None = None,
    ):
        self.task = task if isinstance(task, TaskType) else TaskType(task)
        self.dataset_name = dataset_name
        self.task_hint = task_hint
        self.config = config if isinstance(config, DatasetMonitorConfig) else DatasetMonitorConfig(**(config or {}))
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.run_id = run_id
        self._captures: list[StageCapture] = []
        self._errors: list[dict[str, Any]] = []

    def record_stage(
        self,
        stage: PipelineStage | str,
        *,
        split: str,
        data: Any,
        labels: Any = None,
        metadata: Optional[dict[str, Any]] = None,
        latency_ms: float | None = None,
        parent_stage: PipelineStage | str | None = None,
        sample_limit: int | None = None,
    ) -> "DatasetMonitorBuilder":
        normalized_stage = _normalize_stage(stage)
        parent = _normalize_stage(parent_stage) if parent_stage is not None else None
        self._captures.append(
            StageCapture(
                stage=normalized_stage,
                split=str(split),
                data=data,
                labels=labels,
                metadata=dict(metadata or {}),
                latency_ms=latency_ms,
                parent_stage=parent,
                sample_limit=sample_limit,
            )
        )
        return self

    def _build_stage_records(self) -> list[StageRecord]:
        records: list[StageRecord] = []
        for capture in self._captures:
            sample_limit = capture.sample_limit or self.config.sample_budget_per_split
            adapted = adapt_input(capture.data, capture.labels, limit=sample_limit)
            observations = []
            modality_counts = {}
            label_available = 0
            missing_rates = []
            schema_fields = set()
            for adapted_sample in adapted.records:
                try:
                    observation = inspect_sample(
                        adapted_sample.sample,
                        index=adapted_sample.index,
                        split=capture.split,
                        label=adapted_sample.label,
                        task=self.task,
                        task_hint=self.task_hint,
                        signature_dims=self.config.approximate_signature_dims,
                    )
                    observations.append(observation)
                    modality_counts[observation.modality.value] = modality_counts.get(observation.modality.value, 0) + 1
                    if observation.label is not None:
                        label_available += 1
                    missing_rates.append(observation.missing_rate)
                    schema_fields.update(observation.schema_fields)
                except Exception as exc:
                    self._errors.append(
                        {
                            "stage": capture.stage.value,
                            "split": capture.split,
                            "type": "inspection_error",
                            "message": str(exc),
                        }
                    )

            source_sample_count = capture.metadata.get("source_sample_count")
            try:
                source_sample_count = int(source_sample_count) if source_sample_count is not None else None
            except (TypeError, ValueError):
                source_sample_count = None

            total_count = adapted.total_count if adapted.total_count is not None else len(observations)
            if source_sample_count is not None:
                total_count = max(total_count, source_sample_count, len(observations))
            modality_total = max(sum(modality_counts.values()), 1)
            snapshot = PipelineStageSnapshot(
                stage=capture.stage,
                stage_label=PIPELINE_STAGE_LABELS[capture.stage],
                stage_index=PIPELINE_STAGE_ORDER.index(capture.stage),
                split=capture.split,
                parent_stage=capture.parent_stage,
                sample_count=total_count,
                observed_sample_count=len(observations),
                null_missing_rate=round(float(sum(missing_rates) / max(len(missing_rates), 1)), 6)
                if observations
                else None,
                label_availability="available"
                if label_available == len(observations) and observations
                else ("partial" if label_available else "missing"),
                modality_metadata={
                    "modality_proportions": {
                        key: round(value / modality_total, 4) for key, value in sorted(modality_counts.items())
                    },
                    "modalities": sorted(modality_counts.keys()),
                    "schema_fields": sorted(schema_fields),
                },
                latency_ms=capture.latency_ms,
                metrics={
                    "adapter_source_type": adapted.source_type,
                    "adapter_errors": adapted.errors,
                    "source_sample_count": source_sample_count,
                },
                notes=[f"Observed {len(observations)} samples from {adapted.source_type}."],
            )
            records.append(
                StageRecord(
                    stage=capture.stage,
                    split=capture.split,
                    snapshot=snapshot,
                    observations=observations,
                    metadata=capture.metadata,
                    adapter_errors=adapted.errors,
                )
            )
        return enrich_stage_snapshots(records)

    def _build_dashboard(
        self, report: DatasetMonitorReport, stage_records: list[StageRecord], latest_by_split: dict[str, StageRecord]
    ) -> DashboardViewModel:
        checks = report.checks
        stage_cards = []
        for stage in PIPELINE_STAGE_ORDER:
            matching_records = [record for record in stage_records if record.stage == stage]
            if not matching_records:
                stage_cards.append(
                    {
                        "stage": stage.value,
                        "title": PIPELINE_STAGE_LABELS[stage],
                        "status": CheckStatus.UNKNOWN.value,
                        "sample_count": None,
                        "splits": [],
                    }
                )
                continue
            sample_count = sum((record.snapshot.sample_count or 0) for record in matching_records)
            statuses = [record.snapshot.status for record in matching_records]
            if CheckStatus.FAILED in statuses:
                status = CheckStatus.FAILED.value
            elif CheckStatus.WARNING in statuses:
                status = CheckStatus.WARNING.value
            elif all(item == CheckStatus.PASSED for item in statuses):
                status = CheckStatus.PASSED.value
            else:
                status = CheckStatus.UNKNOWN.value
            stage_cards.append(
                {
                    "stage": stage.value,
                    "title": PIPELINE_STAGE_LABELS[stage],
                    "status": status,
                    "sample_count": sample_count if sample_count > 0 else None,
                    "splits": sorted(record.split for record in matching_records),
                }
            )

        composition_panels: list[DashboardPanel] = []
        for split, slice_model in report.composition.latest_by_split.items():
            composition_panels.append(
                DashboardPanel(
                    id=f"modality-{split}",
                    type="modality-breakdown",
                    title=f"{split.title()} Modality Mix",
                    split=split,
                    category="composition",
                    data={"series": slice_model.modality_proportions},
                )
            )
            if slice_model.class_distribution:
                composition_panels.append(
                    DashboardPanel(
                        id=f"class-{split}",
                        type="class-distribution",
                        title=f"{split.title()} Class Distribution",
                        split=split,
                        category="composition",
                        data={"series": slice_model.class_distribution},
                    )
                )
            elif slice_model.regression_target_distribution.get("available"):
                composition_panels.append(
                    DashboardPanel(
                        id=f"regression-{split}",
                        type="histogram",
                        title=f"{split.title()} Target Distribution",
                        split=split,
                        category="composition",
                        data=slice_model.regression_target_distribution,
                    )
                )
            if slice_model.sequence_length_distribution.get("available"):
                composition_panels.append(
                    DashboardPanel(
                        id=f"sequence-{split}",
                        type="histogram",
                        title=f"{split.title()} Sequence Lengths",
                        split=split,
                        category="composition",
                        data=slice_model.sequence_length_distribution,
                    )
                )
            if slice_model.image_size_distribution.get("available"):
                composition_panels.append(
                    DashboardPanel(
                        id=f"image-height-{split}",
                        type="histogram",
                        title=f"{split.title()} Image Heights",
                        split=split,
                        category="composition",
                        data=slice_model.image_size_distribution.get("heights", {}),
                    )
                )
                composition_panels.append(
                    DashboardPanel(
                        id=f"image-width-{split}",
                        type="histogram",
                        title=f"{split.title()} Image Widths",
                        split=split,
                        category="composition",
                        data=slice_model.image_size_distribution.get("widths", {}),
                    )
                )
            if slice_model.image_aspect_ratio_distribution.get("available"):
                composition_panels.append(
                    DashboardPanel(
                        id=f"image-aspect-{split}",
                        type="histogram",
                        title=f"{split.title()} Image Aspect Ratios",
                        split=split,
                        category="composition",
                        data=slice_model.image_aspect_ratio_distribution,
                    )
                )
            if slice_model.outlier_features:
                composition_panels.append(
                    DashboardPanel(
                        id=f"outliers-{split}",
                        type="outlier-table",
                        title=f"{split.title()} Top Outlier Features",
                        split=split,
                        category="composition",
                        data={"rows": slice_model.outlier_features},
                    )
                )

        split_panels = [
            DashboardPanel(
                id=f"split-{index}",
                type="split-comparison",
                title=f"{comparison.split_a.title()} vs {comparison.split_b.title()}",
                category="comparison",
                data=comparison.model_dump(mode="json"),
            )
            for index, comparison in enumerate(report.split_comparisons)
        ]

        leakage_checks = [
            check.model_dump(mode="json")
            for check in checks
            if check.category == CheckCategory.LEAKAGE or check.category.value == CheckCategory.LEAKAGE.value
        ]
        recommendation_cards = [item.model_dump(mode="json") for item in report.recommendations]
        statuses = [check.status for check in checks]
        overall_status = (
            CheckStatus.FAILED
            if CheckStatus.FAILED in statuses
            else CheckStatus.WARNING
            if CheckStatus.WARNING in statuses
            else CheckStatus.PASSED
            if statuses and all(status == CheckStatus.PASSED for status in statuses)
            else CheckStatus.UNKNOWN
        )

        return DashboardViewModel(
            summary={
                "overall_status": overall_status.value,
                "total_checks": len(checks),
                "failed_checks": sum(1 for check in checks if check.status == CheckStatus.FAILED),
                "warning_checks": sum(1 for check in checks if check.status == CheckStatus.WARNING),
                "unknown_checks": sum(1 for check in checks if check.status == CheckStatus.UNKNOWN),
                "recorded_stage_count": len(report.pipeline.get("snapshots", [])),
            },
            stage_cards=stage_cards,
            composition_panels=composition_panels,
            split_comparison_panels=split_panels,
            leakage_checks=leakage_checks,
            recommendation_cards=recommendation_cards,
            filters={
                "splits": sorted(report.metadata.available_splits),
                "stages": [stage.value for stage in PIPELINE_STAGE_ORDER],
                "categories": sorted({check.category.value for check in checks}),
                "severities": sorted({check.severity.value for check in checks}),
                "statuses": sorted({check.status.value for check in checks}),
                "modalities": sorted(modality.value for modality in report.metadata.available_modalities),
            },
        )

    def _render_summary(self, report: DatasetMonitorReport) -> str:
        lines = []
        lines.append("Dataset & Pipeline Monitor")
        lines.append("=" * 60)
        lines.append(f"Task: {report.metadata.task.value}")
        if report.metadata.dataset_name:
            lines.append(f"Dataset: {report.metadata.dataset_name}")
        lines.append(f"Recorded stages: {report.metadata.recorded_stage_count}")
        lines.append(f"Available splits: {', '.join(report.metadata.available_splits) or '-'}")
        lines.append(
            f"Available modalities: {', '.join(modality.value for modality in report.metadata.available_modalities) or '-'}"
        )
        lines.append("")
        lines.append("Pipeline")
        lines.append("-" * 60)
        for snapshot in report.pipeline.get("snapshots", []):
            stage_name = snapshot["stage_label"]
            split = snapshot["split"]
            status = snapshot["status"]
            sample_count = snapshot.get("sample_count")
            lines.append(
                f"{stage_name:<24} {split:<12} status={status:<8} sample_count={sample_count if sample_count is not None else '-'}"
            )
        lines.append("")
        lines.append("Checks")
        lines.append("-" * 60)
        for check in report.checks:
            lines.append(f"[{check.status.value.upper():<7}] [{check.severity.value:<8}] {check.name}")
            lines.append(f"  {check.summary}")
            if check.recommendation:
                lines.append(f"  Recommendation: {check.recommendation}")
        if report.recommendations:
            lines.append("")
            lines.append("Recommended next steps")
            lines.append("-" * 60)
            for item in report.recommendations[:5]:
                lines.append(f"[{item.severity.value}] {item.title}: {item.summary}")
        return "\n".join(lines)

    def _save_report(self, report: DatasetMonitorReport) -> None:
        if self.run_dir is None:
            return
        analysis_dir = self.run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(analysis_dir / "dataset_monitor.json", "w") as handle:
            json.dump(report.model_dump(mode="json"), handle, indent=2)
        with open(analysis_dir / "dataset_monitor_summary.txt", "w") as handle:
            handle.write(report.summary_text)

    def finalize(self, save: bool = True) -> DatasetMonitorReport:
        stage_records = self._build_stage_records()
        latest_by_split = {}
        for record in stage_records:
            latest_by_split[record.split] = record

        slices = [
            build_composition_slice(record.stage, record.split, record.observations, self.config, record.metadata)
            for record in stage_records
        ]
        latest_slices = {}
        for record, slice_model in zip(stage_records, slices):
            latest_slices[record.split] = slice_model

        overall_slice = CompositionSlice(
            stage=PipelineStage.SPLITTING if not stage_records else stage_records[-1].stage,
            split="overall",
            sample_count=sum(slice_model.sample_count for slice_model in latest_slices.values()),
            modality_proportions={},
        )
        if latest_slices:
            aggregate_modalities = {}
            total_samples = max(sum(slice_model.sample_count for slice_model in latest_slices.values()), 1)
            aggregate_classes = {}
            for slice_model in latest_slices.values():
                for modality, proportion in slice_model.modality_proportions.items():
                    aggregate_modalities[modality] = (
                        aggregate_modalities.get(modality, 0.0) + proportion * slice_model.sample_count
                    )
                for label, proportion in slice_model.class_distribution.items():
                    aggregate_classes[label] = aggregate_classes.get(label, 0.0) + proportion * slice_model.sample_count
            overall_slice.modality_proportions = {
                key: round(value / total_samples, 4) for key, value in aggregate_modalities.items()
            }
            overall_slice.class_distribution = {
                key: round(value / total_samples, 4) for key, value in aggregate_classes.items()
            }

        checks = run_leakage_checks(stage_records, self.task, self.config)
        recommendations = build_recommendations(checks, self.config)
        metadata = DatasetMonitorMetadata(
            run_id=self.run_id,
            dataset_name=self.dataset_name,
            task=self.task,
            task_hint=self.task_hint,
            available_splits=sorted(latest_by_split.keys()),
            available_modalities=sorted(
                {
                    detect_modality(obs.sample, task=self.task, task_hint=self.task_hint)
                    for record in stage_records
                    for obs in record.observations
                },
                key=lambda item: item.value,
            ),
            recorded_stage_count=len(stage_records),
            total_checks=len(checks),
            overall_status=(
                CheckStatus.FAILED
                if any(check.status == CheckStatus.FAILED for check in checks)
                else CheckStatus.WARNING
                if any(check.status == CheckStatus.WARNING for check in checks)
                else CheckStatus.PASSED
                if checks and all(check.status == CheckStatus.PASSED for check in checks)
                else CheckStatus.UNKNOWN
            ),
        )
        report = DatasetMonitorReport(
            metadata=metadata,
            pipeline={
                "stage_order": [stage.value for stage in PIPELINE_STAGE_ORDER],
                "snapshots": [record.snapshot.model_dump(mode="json") for record in stage_records],
            },
            composition=CompositionSummary(overall=overall_slice, latest_by_split=latest_slices, slices=slices),
            split_comparisons=build_split_comparisons(latest_by_split, self.config),
            checks=checks,
            recommendations=recommendations,
            errors=self._errors
            + [
                {"stage": record.stage.value, "split": record.split, "type": "adapter_error", "message": error}
                for record in stage_records
                for error in record.adapter_errors
            ],
        )
        report.summary_text = self._render_summary(report)
        report.dashboard = self._build_dashboard(report, stage_records, latest_by_split)
        if save:
            self._save_report(report)
        return report


def load_dataset_monitor_report(run_dir: Path) -> DatasetMonitorReport | None:
    report_path = Path(run_dir) / "analysis" / "dataset_monitor.json"
    if not report_path.exists():
        return None
    with open(report_path) as handle:
        data = json.load(handle)
    return DatasetMonitorReport.model_validate(data)
