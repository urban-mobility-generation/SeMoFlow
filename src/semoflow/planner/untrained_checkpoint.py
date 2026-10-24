from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from semoflow.config.schemas.planner_config import (
    OccurrencePlannerTrainingOptions,
)
from semoflow.planner.artifacts import (
    load_occurrence_planner_checkpoint,
    load_occurrence_planner_dataset,
    save_occurrence_planner_checkpoint,
)
from semoflow.planner.legality import LegalPrefixIndex
from semoflow.planner.model import OccurrenceARPlanner
from semoflow.planner.training import split_nll_metrics
from semoflow.planner.warm_start import (
    PlannerWarmStart,
    apply_planner_warm_start,
)
from semoflow.utils.serialization import write_json


def publish_untrained_warm_start(
    *,
    dataset_path: Path,
    initialize_from: Path,
    output_dir: Path,
    batch_size: int = 1024,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 0,
    force: bool = False,
) -> dict[str, Any]:
    checkpoint_path = output_dir / "best.pt"
    warm_start_path = output_dir / "warm_start.json"
    if not force:
        for path in (checkpoint_path, warm_start_path):
            if path.exists():
                raise FileExistsError(
                    f"refusing to overwrite existing artifact: {path}"
                )

    torch.manual_seed(seed)
    dataset = load_occurrence_planner_dataset(dataset_path)
    data = dataset.config
    config = replace(
        load_occurrence_planner_checkpoint(initialize_from).config,
        codebook_size=data.codebook_size,
        gap_vocab=data.gap_vocab,
        prefix_depth=data.prefix_depth,
        max_occurrences=data.max_occurrences,
        context_dim=data.context_dim,
        context_schema=data.context_schema,
        city_conditioning=data.city_conditioning,
        gap_codec_version=data.gap_codec_version,
        gap_bucket_edges_minutes=data.gap_bucket_edges_minutes,
        gap_duration_minutes=data.gap_duration_minutes,
        observation_window_minutes=data.observation_window_minutes,
    )
    if config.use_tokenizer_latent and dataset.prefix_codebooks is None:
        raise ValueError(
            "the source uses the tokenizer latent; the target dataset must be "
            "built with --codebooks"
        )
    if config.use_distance_prior and dataset.legal_prefix_centroids is None:
        raise ValueError(
            "the source uses the distance prior; the target dataset must be "
            "built with --places coordinates"
        )
    prefix_codebooks = (
        dataset.prefix_codebooks if config.use_tokenizer_latent else None
    )
    centroids = (
        dataset.legal_prefix_centroids if config.use_distance_prior else None
    )
    centroid_available = (
        dataset.legal_prefix_centroid_available
        if config.use_distance_prior
        else None
    )
    model = OccurrenceARPlanner(
        config,
        dataset.legal_prefixes,
        catalog_city_ids=dataset.catalog_city_ids,
        city_legal_prefix_mask=dataset.city_legal_prefix_mask,
        prefix_codebooks=prefix_codebooks,
        legal_prefix_centroids=centroids,
        legal_prefix_centroid_available=centroid_available,
    )
    report = apply_planner_warm_start(
        model, PlannerWarmStart(checkpoint=initialize_from)
    )

    target = torch.device(device)
    model = model.to(target)
    validation_metric = split_nll_metrics(
        model,
        dataset.validation,
        config=config,
        legality=LegalPrefixIndex(
            dataset.legal_prefixes,
            codebook_size=config.codebook_size,
            prefix_depth=config.prefix_depth,
        ),
        batch_size=batch_size,
        device=target,
    ).total
    model = model.to("cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_occurrence_planner_checkpoint(
        checkpoint_path,
        config=config,
        legal_prefixes=dataset.legal_prefixes,
        catalog_city_ids=dataset.catalog_city_ids,
        city_legal_prefix_mask=dataset.city_legal_prefix_mask,
        model_state=model.state_dict(),
        epoch=1,
        validation_metric=validation_metric,
        seed=seed,
        options=OccurrencePlannerTrainingOptions(
            epochs=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            length_weight=1.0,
            gap_weight=1.0,
            prefix_weight=1.0,
            label_smoothing=0.0,
            device="cpu",
        ),
        prefix_codebooks=prefix_codebooks,
        legal_prefix_centroids=centroids,
        legal_prefix_centroid_available=centroid_available,
        force=force,
    )

    payload = {
        "artifact_kind": "planner_warm_start_untrained",
        "cell": "P0",
        "trained_epochs": 0,
        "validation_nll_without_adaptation": validation_metric,
        "options_epochs_is_schema_minimum_not_a_run": True,
        "checkpoint_epoch_field_is_schema_floor_not_a_trained_epoch": True,
        "output_checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        **report.to_dict(),
    }
    write_json(warm_start_path, payload)
    return payload


__all__ = ["publish_untrained_warm_start"]
