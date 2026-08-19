"""Stable internal contracts for detector construction.

These types are internal only.  They make the workflow less dependent on file
name conventions and tuple ordering while keeping public artifact names
explicit and reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


MODEL_CONFIG_NAME = "model_candidates.json"
DRIVER_LOG_NAME = "detector_build_workflow.log.txt"
FEATURE_INVENTORY_NAME = "feature_inventory.json"
README_NAME = "README.md"
RUN_COMMANDS_NAME = "RUN_COMMANDS.md"
FEATURE_SEMANTICS_DRAFT_NAME = ".feature_semantics.json"
MODEL_ADVICE_DRAFT_NAME = ".model_advice.json"
FEATURE_IMPLEMENTATION_DRAFT_NAME = ".feature_implementation.py"

class DetectorTarget(str, Enum):
    """The scientific label contract implemented by the generated runtime."""

    MERGE = "merge_detection"
    SPLIT = "split_detection"
    LEGACY_UNSPECIFIED = "legacy_unspecified"


@dataclass(frozen=True)
class TargetSpec:
    """Driver-owned vocabulary and public file contract for one detector task.

    Feature agents never choose these values.  Keeping them in one typed object
    prevents a split run from silently inheriting merge labels, filenames, or
    report terminology merely because a downstream helper was overlooked.
    """

    target: DetectorTarget
    detector_name: str
    output_prefix: str
    row_unit: str
    row_unit_plural: str
    label_name: str
    positive_name: str
    negative_name: str
    # Display plurals and CSV identity columns are part of the same public
    # vocabulary: the result-analysis driver derives its evidence/report
    # wording and CSV column contract from here instead of a local copy.
    positive_plural: str
    negative_plural: str
    identity_columns: tuple[str, ...]
    score_prefix: str
    accumulator_name: str
    candidate_radius_um: float | None = None


_TARGET_SPECS = {
    DetectorTarget.MERGE: TargetSpec(
        target=DetectorTarget.MERGE,
        detector_name="merge_site_detector.py",
        output_prefix="merge_detector",
        row_unit="segment",
        row_unit_plural="segments",
        label_name="is_merge",
        positive_name="merge",
        negative_name="clean",
        positive_plural="merges",
        negative_plural="clean segments",
        identity_columns=("segment_id",),
        score_prefix="merge_probability",
        accumulator_name="SegmentAccumulator",
    ),
    DetectorTarget.SPLIT: TargetSpec(
        target=DetectorTarget.SPLIT,
        detector_name="split_site_detector.py",
        output_prefix="split_detector",
        row_unit="candidate segment pair",
        row_unit_plural="candidate segment pairs",
        label_name="is_split",
        positive_name="split",
        negative_name="non-split candidate",
        positive_plural="splits",
        negative_plural="non-split candidates",
        identity_columns=("candidate_id", "segment_id_a", "segment_id_b"),
        score_prefix="split_probability",
        accumulator_name="FeatureAccumulator",
        # The first supported split run uses nearby-endpoint candidate radii up
        # to 30 um. Candidate generation is driver-owned and remains fixed when
        # hypotheses are later excluded.
        candidate_radius_um=30.0,
    ),
}


def target_spec(target: DetectorTarget) -> TargetSpec:
    """Resolve legacy unnamed runs to merge without weakening named-run checks."""
    if target is DetectorTarget.LEGACY_UNSPECIFIED:
        target = DetectorTarget.MERGE
    try:
        return _TARGET_SPECS[target]
    except KeyError as exc:  # defensive if the enum grows without a contract
        raise ValueError(f"No detector target specification for {target!r}.") from exc


# Compatibility aliases for callers and old tests that intentionally assert the
# established merge public contract.  New workflow code uses ``target_spec``.
DETECTOR_NAME = _TARGET_SPECS[DetectorTarget.MERGE].detector_name
# This tuple is also an output-contract assertion for the established merge
# workflow: refactoring must not silently rename/remove required deliverables.
BUILD_ARTIFACT_NAMES = (
    FEATURE_INVENTORY_NAME,
    MODEL_CONFIG_NAME,
    DETECTOR_NAME,
    README_NAME,
    RUN_COMMANDS_NAME,
    DRIVER_LOG_NAME,
)


def build_artifact_names(target: DetectorTarget) -> tuple[str, ...]:
    spec = target_spec(target)
    return (
        FEATURE_INVENTORY_NAME,
        MODEL_CONFIG_NAME,
        spec.detector_name,
        README_NAME,
        RUN_COMMANDS_NAME,
        DRIVER_LOG_NAME,
    )


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
