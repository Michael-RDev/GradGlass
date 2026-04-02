from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class StrEnum(str, Enum):
    pass


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RANKING = "ranking"
    LANGUAGE_MODELING = "language_modeling"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    TABULAR = "tabular"
    SEQUENCE_TIME_SERIES = "sequence_time_series"
    UNKNOWN = "unknown"


class ModalityType(StrEnum):
    TABULAR = "tabular"
    TEXT = "text"
    TOKEN_SEQUENCE = "token_sequence"
    IMAGE = "image"
    AUDIO = "audio"
    TENSOR = "tensor"
    MULTIMODAL = "multimodal"
    UNKNOWN = "unknown"


class PipelineStage(StrEnum):
    RAW_DATA = "raw_data"
    CLEANING = "cleaning"
    AUGMENTATION = "augmentation"
    TOKENIZATION = "tokenization"
    FEATURE_EXTRACTION = "feature_extraction"
    SPLITTING = "splitting"
    LOADER = "loader"
    BATCH_COLLATION = "batch_collation"


class CheckSeverity(StrEnum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class CheckStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    UNKNOWN = "unknown"


class CheckCategory(StrEnum):
    PIPELINE = "pipeline"
    COMPOSITION = "composition"
    LEAKAGE = "leakage"
    QUALITY = "quality"
    PREPROCESSING = "preprocessing"
    CONSISTENCY = "consistency"
    SCHEMA = "schema"


PIPELINE_STAGE_ORDER = [
    PipelineStage.RAW_DATA,
    PipelineStage.CLEANING,
    PipelineStage.AUGMENTATION,
    PipelineStage.TOKENIZATION,
    PipelineStage.FEATURE_EXTRACTION,
    PipelineStage.SPLITTING,
    PipelineStage.LOADER,
    PipelineStage.BATCH_COLLATION,
]

PIPELINE_STAGE_LABELS = {
    PipelineStage.RAW_DATA: "Raw Data",
    PipelineStage.CLEANING: "Cleaning",
    PipelineStage.AUGMENTATION: "Augmentation",
    PipelineStage.TOKENIZATION: "Tokenization",
    PipelineStage.FEATURE_EXTRACTION: "Feature Extraction",
    PipelineStage.SPLITTING: "Splitting",
    PipelineStage.LOADER: "Loader / Dataloader",
    PipelineStage.BATCH_COLLATION: "Batch Collation",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DatasetMonitorConfig(BaseModel):
    sample_budget_per_split: int = 5000
    pairwise_candidate_budget: int = 50000
    evidence_limit: int = 100
    top_feature_limit: int = 10
    histogram_bins: int = 20
    approximate_signature_dims: int = 16
    near_duplicate_similarity_threshold: float = 0.94
    leakage_correlation_threshold: float = 0.95
    random_seed: int = 0


class PipelineStageSnapshot(BaseModel):
    stage: PipelineStage
    stage_label: str
    stage_index: int
    split: str
    parent_stage: Optional[PipelineStage] = None
    sample_count: Optional[int] = None
    observed_sample_count: int = 0
    sample_coverage: Optional[float] = None
    dropped_samples: Optional[int] = None
    added_samples: Optional[int] = None
    null_missing_rate: Optional[float] = None
    schema_changes: list[dict[str, Any]] = Field(default_factory=list)
    label_availability: str = "unknown"
    modality_metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[float] = None
    status: CheckStatus = CheckStatus.UNKNOWN
    metrics: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CompositionSlice(BaseModel):
    stage: PipelineStage
    split: str
    sample_count: int = 0
    modality_proportions: dict[str, float] = Field(default_factory=dict)
    class_distribution: dict[str, float] = Field(default_factory=dict)
    regression_target_distribution: dict[str, Any] = Field(default_factory=dict)
    sequence_length_distribution: dict[str, Any] = Field(default_factory=dict)
    image_size_distribution: dict[str, Any] = Field(default_factory=dict)
    image_aspect_ratio_distribution: dict[str, Any] = Field(default_factory=dict)
    audio_duration_distribution: dict[str, Any] = Field(default_factory=dict)
    audio_sample_rates: dict[str, int] = Field(default_factory=dict)
    missingness_patterns: dict[str, Any] = Field(default_factory=dict)
    categorical_cardinality: dict[str, int] = Field(default_factory=dict)
    numerical_feature_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)
    provenance: dict[str, float] = Field(default_factory=dict)
    outlier_features: list[dict[str, Any]] = Field(default_factory=list)


class CompositionSummary(BaseModel):
    overall: CompositionSlice
    latest_by_split: dict[str, CompositionSlice] = Field(default_factory=dict)
    slices: list[CompositionSlice] = Field(default_factory=list)


class SplitComparisonSummary(BaseModel):
    split_a: str
    split_b: str
    label_distribution_diff: dict[str, Any] = Field(default_factory=dict)
    numeric_drift_summary: dict[str, Any] = Field(default_factory=dict)
    modality_diff: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""


class MonitorCheckResult(BaseModel):
    name: str
    category: CheckCategory
    severity: CheckSeverity
    status: CheckStatus
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)
    recommendation: Optional[str] = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class RecommendationItem(BaseModel):
    title: str
    severity: CheckSeverity
    summary: str
    confidence: float = 0.5
    affected_splits: list[str] = Field(default_factory=list)
    related_checks: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    score: float = 0.0


class DashboardPanel(BaseModel):
    id: str
    type: str
    title: str
    stage: Optional[str] = None
    split: Optional[str] = None
    category: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class DashboardViewModel(BaseModel):
    summary: dict[str, Any] = Field(default_factory=dict)
    stage_cards: list[dict[str, Any]] = Field(default_factory=list)
    composition_panels: list[DashboardPanel] = Field(default_factory=list)
    split_comparison_panels: list[DashboardPanel] = Field(default_factory=list)
    leakage_checks: list[dict[str, Any]] = Field(default_factory=list)
    recommendation_cards: list[dict[str, Any]] = Field(default_factory=list)
    filters: dict[str, list[str]] = Field(default_factory=dict)


class DatasetMonitorMetadata(BaseModel):
    run_id: Optional[str] = None
    dataset_name: Optional[str] = None
    task: TaskType = TaskType.UNKNOWN
    task_hint: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    available_splits: list[str] = Field(default_factory=list)
    available_modalities: list[ModalityType] = Field(default_factory=list)
    recorded_stage_count: int = 0
    total_checks: int = 0
    overall_status: CheckStatus = CheckStatus.UNKNOWN


class DatasetMonitorReport(BaseModel):
    metadata: DatasetMonitorMetadata
    pipeline: dict[str, Any] = Field(default_factory=dict)
    composition: CompositionSummary
    split_comparisons: list[SplitComparisonSummary] = Field(default_factory=list)
    checks: list[MonitorCheckResult] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    dashboard: DashboardViewModel = Field(default_factory=DashboardViewModel)
    summary_text: str = ""
    errors: list[dict[str, Any]] = Field(default_factory=list)
