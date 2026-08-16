"""Stable internal contracts for detector construction.

These types are internal only.  They make the workflow less dependent on file
name conventions and tuple ordering while keeping public artifact names
explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


DETECTOR_NAME = "merge_site_detector.py"
MODEL_CONFIG_NAME = "model_candidates.json"
DRIVER_LOG_NAME = "detector_build_workflow.log.txt"
FEATURE_INVENTORY_NAME = "feature_inventory.json"
README_NAME = "README.md"
RUN_COMMANDS_NAME = "RUN_COMMANDS.md"
FEATURE_SEMANTICS_DRAFT_NAME = ".feature_semantics.json"
MODEL_ADVICE_DRAFT_NAME = ".model_advice.json"
FEATURE_IMPLEMENTATION_DRAFT_NAME = ".feature_implementation.py"

# This tuple is also an output-contract assertion: refactoring the internals
# must not silently rename, remove, or add required build deliverables.
BUILD_ARTIFACT_NAMES = (
    FEATURE_INVENTORY_NAME,
    MODEL_CONFIG_NAME,
    DETECTOR_NAME,
    README_NAME,
    RUN_COMMANDS_NAME,
    DRIVER_LOG_NAME,
)


class DetectorTarget(str, Enum):
    """The scientific label contract implemented by the generated runtime."""

    MERGE = "merge_detection"
    SPLIT = "split_detection"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


@dataclass(frozen=True)
class RunContext:
    """Resolved, cross-checked inputs for one detector-build invocation."""

    target: DetectorTarget
    run_json: Path
    summary_path: Path
    rerun_dir: Path
    fixed_dir: Path | None
    selection_path: Path | None
    corrected_results_path: Path | None
    selected_ids: tuple[int, ...]

    @property
    def predictive(self) -> bool:
        return self.selection_path is not None
