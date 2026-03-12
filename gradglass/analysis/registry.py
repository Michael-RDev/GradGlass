from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Optional
import numpy as np


class TestStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class TestSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TestCategory(str, Enum):
    ARTIFACT = "Artifact & Store Integrity"
    MODEL = "Model Structure"
    METRICS = "Training Metrics"
    CHECKPOINT = "Checkpoint Diff"
    GRADIENT = "Gradient Flow"
    ACTIVATION = "Activations"
    PREDICTION = "Predictions"
    DATA = "Data"
    DISTRIBUTED = "Distributed Training"
    REPRODUCIBILITY = "Reproducibility"


@dataclass
class TestResult:
    id: str
    title: str
    status: TestStatus
    severity: TestSeverity
    category: TestCategory
    details: dict = field(default_factory=dict)
    recommendation: str = ""
    duration_ms: float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status.value,
            "severity": self.severity.value,
            "category": self.category.value,
            "details": self.details,
            "recommendation": self.recommendation,
            "duration_ms": round(self.duration_ms, 2),
        }


@dataclass
class TestContext:
    run_id: str
    run_dir: Path
    store: Any
    metadata: Optional[dict] = None
    metrics: Optional[list] = None
    checkpoints_meta: Optional[list] = None
    architecture: Optional[dict] = None
    gradient_summaries: Optional[list] = None
    activation_stats: Optional[list] = None
    predictions: Optional[list] = None
    distributed_info: Optional[dict] = None
    rank_list: Optional[list] = None

    @property
    def has_checkpoints(self):
        return bool(self.checkpoints_meta)

    @property
    def has_metrics(self):
        return bool(self.metrics)

    @property
    def has_grad_summaries(self):
        return bool(self.gradient_summaries)

    @property
    def has_activations(self):
        return bool(self.activation_stats)

    @property
    def has_predictions(self):
        return bool(self.predictions)

    @property
    def has_architecture(self):
        return self.architecture is not None

    @property
    def is_distributed(self):
        return self.distributed_info is not None or bool(self.rank_list)

    def load_checkpoint(self, step):
        return self.store.load_checkpoint(self.run_id, step)

    def checkpoint_steps(self):
        if not self.checkpoints_meta:
            return []
        return sorted((c["step"] for c in self.checkpoints_meta))


@dataclass
class RegisteredTest:
    id: str
    title: str
    category: TestCategory
    severity: TestSeverity
    description: str
    fn: Callable[[TestContext], TestResult]


class TestRegistry:
    tests: dict[str, RegisteredTest] = {}

    @classmethod
    def register(cls, id, title, category, severity=TestSeverity.MEDIUM, description=""):

        def decorator(fn):
            cls.tests[id] = RegisteredTest(
                id=id, title=title, category=category, severity=severity, description=description, fn=fn
            )
            return fn

        return decorator

    @classmethod
    def get(cls, test_id):
        return cls.tests.get(test_id)

    @classmethod
    def all_tests(cls):
        return dict(cls.tests)

    @classmethod
    def by_category(cls, category):
        return [t for t in cls.tests.values() if t.category == category]

    @classmethod
    def ids(cls):
        return list(cls.tests.keys())


def test(id, title, category, severity=TestSeverity.MEDIUM, description=""):
    return TestRegistry.register(id=id, title=title, category=category, severity=severity, description=description)
