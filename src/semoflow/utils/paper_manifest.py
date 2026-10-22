from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

PAPER_MANIFEST_SCHEMA_VERSION = 1
TRAJECTORY_FID_ESTIMATOR = "countsketch_r4_seed0_v1"
PAPER_MODEL_SEEDS = frozenset({7, 19, 31})
RETIRED_METRIC_NAMES = frozenset({"fid", "latent_fid", "poi_embedding_fid"})

_REVISION_RE = re.compile(r"^[0-9a-f]{7,64}$")


class PaperManifestValidationError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            "paper manifest is ineligible:\n- " + "\n- ".join(errors)
        )


def _mapping(
    payload: Mapping[str, Any], key: str, errors: list[str]
) -> Mapping[str, Any]:
    value = payload.get(key)
    if isinstance(value, Mapping):
        return value
    errors.append(f"{key} must be a mapping")
    return {}


def _nonempty_string(
    payload: Mapping[str, Any], key: str, qualified: str, errors: list[str]
) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{qualified} must be a non-empty string")
        return None
    return value.strip()


def _metric_leaf(name: str) -> str:
    return re.split(r"[./]", name)[-1]


def paper_manifest_errors(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != PAPER_MANIFEST_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {PAPER_MANIFEST_SCHEMA_VERSION}"
        )
    _nonempty_string(payload, "study", "study", errors)

    training = _mapping(payload, "training", errors)
    _nonempty_string(training, "config", "training.config", errors)
    if training.get("bank_split") != "train":
        errors.append("training.bank_split must be exactly 'train'")
    model_seed = training.get("model_seed")
    if isinstance(model_seed, bool) or model_seed not in PAPER_MODEL_SEEDS:
        errors.append(
            f"training.model_seed must be one of {sorted(PAPER_MODEL_SEEDS)}"
        )

    evaluation = _mapping(payload, "evaluation", errors)
    _nonempty_string(evaluation, "config", "evaluation.config", errors)
    eval_split = evaluation.get("bank_split")
    if eval_split not in {"val", "test"}:
        errors.append("evaluation.bank_split must be exactly 'val' or 'test'")
    sampling_seed = evaluation.get("sampling_seed")
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        errors.append("evaluation.sampling_seed must be an integer")
    revision = _nonempty_string(
        evaluation,
        "code_revision",
        "evaluation.code_revision",
        errors,
    )
    if revision is not None and not _REVISION_RE.fullmatch(revision.lower()):
        errors.append(
            "evaluation.code_revision must be a 7-64 character hex revision"
        )

    freeze = _mapping(payload, "freeze", errors)
    if eval_split == "test":
        for field in (
            "method_frozen",
            "analysis_frozen",
            "test_access_ledger_recorded",
        ):
            if freeze.get(field) is not True:
                errors.append(
                    f"freeze.{field} must be true before test access"
                )

    artifacts = _mapping(payload, "artifacts", errors)
    if not artifacts:
        errors.append("artifacts must contain at least one input")
    for name, raw_artifact in artifacts.items():
        qualified = f"artifacts.{name}"
        if not isinstance(raw_artifact, Mapping):
            errors.append(f"{qualified} must be a mapping")
            continue
        _nonempty_string(raw_artifact, "path", f"{qualified}.path", errors)

    metrics = _mapping(payload, "metrics", errors)
    raw_names = metrics.get("names")
    names: list[str] = []
    if not isinstance(raw_names, list) or not raw_names:
        errors.append("metrics.names must be a non-empty list")
    else:
        for index, name in enumerate(raw_names):
            if not isinstance(name, str) or not name.strip():
                errors.append(
                    f"metrics.names[{index}] must be a non-empty string"
                )
                continue
            names.append(name.strip())
    retired = sorted(
        name for name in names if _metric_leaf(name) in RETIRED_METRIC_NAMES
    )
    if retired:
        errors.append(
            "retired metric name(s) are forbidden: " + ", ".join(retired)
        )
    if (
        any(_metric_leaf(name) == "trajectory_fid" for name in names)
        and metrics.get("trajectory_fid_estimator") != TRAJECTORY_FID_ESTIMATOR
    ):
        errors.append(
            "metrics.trajectory_fid_estimator must be "
            f"'{TRAJECTORY_FID_ESTIMATOR}'"
        )

    uncertainty = _mapping(metrics, "uncertainty", errors)
    if uncertainty.get("resampling_unit") != "device_cluster":
        errors.append(
            "metrics.uncertainty.resampling_unit must be 'device_cluster'"
        )
    bootstrap_resamples = uncertainty.get("bootstrap_resamples")
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples < 1000
    ):
        errors.append(
            "metrics.uncertainty.bootstrap_resamples must be at least 1000"
        )
    n_clusters = uncertainty.get("n_clusters")
    if (
        isinstance(n_clusters, bool)
        or not isinstance(n_clusters, int)
        or n_clusters <= 0
    ):
        errors.append("metrics.uncertainty.n_clusters must be positive")
    return errors


def validate_paper_manifest(payload: Mapping[str, Any]) -> None:
    errors = paper_manifest_errors(payload)
    if errors:
        raise PaperManifestValidationError(errors)


__all__ = [
    "PAPER_MANIFEST_SCHEMA_VERSION",
    "PAPER_MODEL_SEEDS",
    "RETIRED_METRIC_NAMES",
    "TRAJECTORY_FID_ESTIMATOR",
    "PaperManifestValidationError",
    "paper_manifest_errors",
    "validate_paper_manifest",
]
