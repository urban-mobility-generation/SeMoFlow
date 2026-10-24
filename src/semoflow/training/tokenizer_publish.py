from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from semoflow.config import ExperimentConfig
from semoflow.config.artifacts import TokenizerArtifacts
from semoflow.tokenization.features import load_graph_embeddings
from semoflow.tokenization.shared_local import (
    semantic_id_lookup_fingerprint,
    shared_codebook_fingerprint,
)
from semoflow.utils.provenance import (
    ClosureComponent,
    ProvenanceClosure,
    load_closure,
    write_closure,
)
from semoflow.utils.run_metadata import write_run_manifest

logger = logging.getLogger(__name__)


def _attach_catalog_city_ids(
    lookup: pl.DataFrame,
    places_index: pl.DataFrame,
) -> pl.DataFrame:
    if "city_id" not in places_index.columns:
        return lookup
    city_lookup = places_index.select("placekey_id", "city_id").unique()
    if city_lookup["placekey_id"].n_unique() != city_lookup.height:
        raise ValueError(
            "tokenizer places index maps a placekey_id to multiple city_ids"
        )
    result = lookup.join(
        city_lookup,
        on="placekey_id",
        how="left",
        validate="m:1",
        maintain_order="left",
    )
    if result["city_id"].null_count() > 0:
        raise ValueError(
            "tokenizer places index must provide city_id for every POI"
        )
    return result


def publish_tokenizer_artifacts(
    cfg: ExperimentConfig,
    artifacts: TokenizerArtifacts,
    *,
    model: Any,
    prepared: Any,
    history: list[Any],
) -> Path | None:
    import torch

    from semoflow.tokenization.decoding.codebooks import SIDCodebookBundle

    tcfg = cfg.tokenizer_config
    stays = prepared.stays
    aug_path = artifacts.augmented_processed_visits

    codes = encode_feature_tensors(model, prepared.tensors, device=tcfg.device)
    id_cols = [
        c
        for c in ("placekey_id", "placekey")
        if c in prepared.poi_features.columns
    ]
    if "placekey_id" not in id_cols:
        raise ValueError("tokenizer POI features must contain placekey_id")
    codes_df = _attach_catalog_city_ids(
        prepared.poi_features.select(id_cols), prepared.places_index
    )
    for i in range(codes.shape[1]):
        codes_df = codes_df.with_columns(
            pl.Series(f"sem_code_{i}", codes[:, i])
        )
    code_cols = [c for c in codes_df.columns if c.startswith("sem_code_")]
    bundle = SIDCodebookBundle.from_tokenizer(model, code_fields=code_cols)
    codebook_fingerprint = shared_codebook_fingerprint(list(bundle.codebooks))
    lookup_fingerprint = semantic_id_lookup_fingerprint(codes_df, code_cols)

    codes_df.write_parquet(artifacts.placekey_lookup)
    write_grounding_artifacts(
        codes_df,
        prepared.places_index,
        code_cols=code_cols,
        grounding_index_path=artifacts.grounding_index,
    )

    if not tcfg.skip_artifact_export:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_columns": prepared.feature_cols,
                "n_codebooks": model.n_codebooks,
                "codebook_size": model.codebook_size,
                "input_dim": len(prepared.feature_cols),
                "graph_dim": prepared.graph_dim,
                "hidden_dim": tcfg.hidden_dim,
                "latent_dim": tcfg.latent_dim,
                "history": history,
                "config": vars(tcfg),
                "codebooks": bundle.codebooks,
                "code_fields": bundle.code_fields,
                "feature_mean": prepared.feature_mean,
                "feature_std": prepared.feature_std,
                "feature_preprocessing": prepared.feature_preprocessing,
                "shared_codebook_fingerprint": codebook_fingerprint,
                "lookup_fingerprint": lookup_fingerprint,
            },
            artifacts.tokenizer_model,
        )
        torch.save(
            {
                "codebooks": bundle.codebooks,
                "code_fields": bundle.code_fields,
                "shared_codebook_fingerprint": codebook_fingerprint,
                "lookup_fingerprint": lookup_fingerprint,
            },
            artifacts.tokenizer_codebooks,
        )

        stays_aug = stays.join(
            codes_df.select(["placekey_id", *code_cols]),
            on="placekey_id",
            how="left",
            maintain_order="left",
        )
        for c in code_cols:
            stays_aug = stays_aug.with_columns(
                pl.col(c).fill_null(0).cast(pl.Int32)
            )
        stays_aug.write_parquet(aug_path)
        logger.info(
            "tokenizer training complete: %s  codebooks=%s",
            aug_path,
            artifacts.tokenizer_codebooks,
        )
        write_tokenizer_provenance(tcfg, artifacts)
        if tcfg.skip_semantic_id_eval:
            logger.info(
                "skip_semantic_id_eval=true; skipping semantic-ID report"
            )
        else:
            run_semantic_id_eval(
                cfg=tcfg,
                artifacts=artifacts,
                places_index=prepared.places_index,
                stays=stays,
            )
        required = (
            "augmented_processed_visits",
            "placekey_lookup",
            "tokenizer_model",
            "tokenizer_codebooks",
            "grounding_index",
        )
        missing = [
            name
            for name in required
            if not artifacts.export_paths()[name].exists()
        ]
        if missing:
            artifacts.write_completion(status="partial", required=required)
            raise RuntimeError(
                "tokenizer publication incomplete; missing "
                + ", ".join(missing)
            )
        artifacts.write_completion(status="complete", required=required)
        return aug_path
    else:
        logger.info(
            "skip_artifact_export=true; skipping heavy artifact writes"
        )
        write_tokenizer_provenance(tcfg, artifacts)
        artifacts.write_completion(
            status="partial",
            required=(),
            skipped=tuple(
                name
                for name in artifacts.export_paths()
                if name != "config.yaml"
            ),
        )
        return None


def write_tokenizer_provenance(
    tcfg: Any, artifacts: TokenizerArtifacts
) -> None:
    closure = ProvenanceClosure(artifact=str(artifacts.run_dir))
    closure.components.append(
        ClosureComponent(
            source=f"tokenizer_text:{tcfg.text_embeddings_path}",
            kind="substrate",
        )
    )
    if tcfg.graphsage_artifact_dir:
        upstream = load_closure(Path(tcfg.graphsage_artifact_dir).resolve())
        if upstream is not None:
            closure.components.extend(upstream.components)
        else:
            closure.components.append(
                ClosureComponent(
                    source=f"graphsage:{tcfg.graphsage_artifact_dir}",
                    kind="behavioral",
                )
            )

    write_closure(closure, artifacts.run_dir)


def run_semantic_id_eval(
    *,
    cfg: Any,
    artifacts: TokenizerArtifacts,
    places_index: pl.DataFrame,
    stays: pl.DataFrame | None = None,
) -> None:
    from semoflow.metrics.semantic_id_baselines import (
        build_simple_baseline_lookups,
        select_baseline_metrics,
    )
    from semoflow.metrics.semantic_id_eval import evaluate_semantic_id_families
    from semoflow.metrics.tokenization import (
        SemanticIDGroundingIndex,
        evaluate_semantic_ids,
        write_semantic_id_eval_report,
    )
    from semoflow.tokenization.decoding.codebooks import (
        extract_codebook_bundle,
    )

    lookup_df = _enrich_sid_lookup_for_eval(
        pl.read_parquet(artifacts.placekey_lookup), places_index
    )
    grounding_idx = SemanticIDGroundingIndex.load(artifacts.grounding_index)
    codebooks = extract_codebook_bundle(
        artifacts.tokenizer_codebooks
    ).codebooks

    cf_embeddings = None
    cf_index = None
    if cfg.graphsage_artifact_dir:
        cf_index, cf_embeddings = load_graph_embeddings(
            Path(cfg.graphsage_artifact_dir)
        )

    latlon_map: dict[str, tuple[float, float]] | None = None
    if {"placekey", "latitude", "longitude"} <= set(places_index.columns):
        latlon_map = {
            str(row["placekey"]): (
                float(row["latitude"]),
                float(row["longitude"]),
            )
            for row in places_index.select(
                ["placekey", "latitude", "longitude"]
            )
            .drop_nulls()
            .iter_rows(named=True)
        }

    result = evaluate_semantic_ids(
        grounding_idx,
        placekey_lookup=lookup_df,
        stays=stays,
        cf_embeddings=cf_embeddings,
        cf_index=cf_index,
        latlon_map=latlon_map,
        codebooks=codebooks,
        codebook_size=cfg.codebook_size,
        k=cfg.eval_k,
        sample_size=cfg.eval_sample_size,
        pair_sample_size=cfg.eval_pair_sample_size,
        geometry_sample_size=cfg.eval_geometry_sample_size,
        geometry_lcp_pair_sample_size=cfg.eval_geometry_lcp_pair_sample_size,
        seed=cfg.seed,
        bootstrap_resamples=cfg.eval_bootstrap_resamples,
        bootstrap_confidence=cfg.eval_bootstrap_confidence,
    )

    baseline_metrics: dict[str, Any] = {}
    baseline_catalog = _enrich_sid_lookup_for_eval(
        grounding_idx.to_frame(), places_index
    )
    text_embeddings = _load_optional_text_embeddings(
        Path(cfg.text_embeddings_path)
    )
    catalog_id = (
        "placekey_id"
        if "placekey_id" in baseline_catalog.columns
        else "placekey"
    )
    catalog_embedding_index = places_index.select(catalog_id).with_row_index(
        "embedding_row"
    )
    graph_embeddings_np = (
        cf_embeddings.detach().cpu().numpy()
        if cf_embeddings is not None
        else None
    )
    simple_lookups = build_simple_baseline_lookups(
        baseline_catalog,
        n_codebooks=result.n_codebooks,
        codebook_size=result.codebook_size,
        seed=cfg.seed,
        text_embeddings=text_embeddings,
        graph_embeddings=graph_embeddings_np,
        text_embedding_index=catalog_embedding_index,
        graph_embedding_index=cf_index,
    )

    def evaluate_baseline(name: str, lookup: pl.DataFrame) -> None:
        enriched = _enrich_sid_lookup_for_eval(lookup, places_index)
        baseline_grounding = _grounding_for_eval_lookup(enriched)
        metrics = evaluate_semantic_id_families(
            baseline_grounding,
            placekey_lookup=enriched,
            stays=stays,
            cf_embeddings=cf_embeddings,
            cf_index=cf_index,
            latlon_map=latlon_map,
            codebook_size=result.codebook_size,
            k=cfg.eval_k,
            sample_size=cfg.eval_sample_size,
            pair_sample_size=cfg.eval_pair_sample_size,
            seed=cfg.seed,
            bootstrap_resamples=cfg.eval_bootstrap_resamples,
            bootstrap_confidence=cfg.eval_bootstrap_confidence,
        )
        baseline_metrics[name] = select_baseline_metrics(metrics)

    for name, baseline_lookup in simple_lookups.items():
        evaluate_baseline(name, baseline_lookup)

    result = dataclasses.replace(result, baseline_metrics=baseline_metrics)

    eval_dir = artifacts.eval_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "semantic_id_eval.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    write_semantic_id_eval_report(result, eval_dir / "semantic_id_eval.md")
    write_run_manifest(
        eval_dir,
        inputs={
            "run_dir": artifacts.run_dir,
            "placekey_lookup": artifacts.placekey_lookup,
            "grounding_index": artifacts.grounding_index,
            "tokenizer_codebooks": artifacts.tokenizer_codebooks,
            "graphsage_artifact_dir": cfg.graphsage_artifact_dir,
        },
        outputs={
            "semantic_id_eval": eval_dir / "semantic_id_eval.json",
            "semantic_id_report": eval_dir / "semantic_id_eval.md",
        },
    )


def _enrich_sid_lookup_for_eval(
    lookup: pl.DataFrame,
    places_index: pl.DataFrame,
) -> pl.DataFrame:
    id_column = (
        "placekey_id" if "placekey_id" in lookup.columns else "placekey"
    )
    if id_column not in places_index.columns:
        return lookup
    metadata_columns = [
        column
        for column in (
            "placekey",
            "top_category",
            "top_cat",
            "geohash_5",
            "latitude",
            "longitude",
            "naics_code",
        )
        if column in places_index.columns and column not in lookup.columns
    ]
    if not metadata_columns:
        return lookup
    return lookup.join(
        places_index.select([id_column, *metadata_columns]).unique(
            subset=[id_column], keep="first"
        ),
        on=id_column,
        how="left",
        maintain_order="left",
    )


def _grounding_for_eval_lookup(lookup: pl.DataFrame) -> Any:
    from semoflow.metrics.tokenization import SemanticIDGroundingIndex
    from semoflow.tokenization.token_space import sid_code_columns

    id_column = (
        "placekey_id" if "placekey_id" in lookup.columns else "placekey"
    )
    code_cols = list(sid_code_columns(lookup.columns))
    payload_columns = [
        column
        for column in (
            "placekey",
            "placekey_id",
            "top_category",
            "top_cat",
            "geohash_5",
            "latitude",
            "longitude",
            "naics_code",
        )
        if column in lookup.columns
    ]
    return SemanticIDGroundingIndex.from_frame(
        lookup,
        key_columns=code_cols,
        id_column=id_column,
        payload_columns=payload_columns,
    )


def _load_optional_text_embeddings(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    values = np.load(path, mmap_mode="r")
    if values.ndim != 2:
        logger.warning(
            "skipping text K-means baseline: expected 2D embeddings"
        )
        return None
    return np.asarray(values)


def encode_feature_tensors(
    model: Any,
    tensors: dict[str, Any],
    *,
    device: str,
) -> Any:
    import torch

    x = tensors["x"]
    parts: list[np.ndarray] = []
    model.to(torch.device(device))
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], 2048):
            batch = {
                k: v[start : start + 2048].to(device)
                for k, v in tensors.items()
            }
            parts.append(model.encode(batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype("int32")


def write_grounding_artifacts(
    codes_df: pl.DataFrame,
    metadata_frame: pl.DataFrame,
    *,
    code_cols: list[str],
    grounding_index_path: Path,
) -> Any:
    from semoflow.metrics.tokenization import SemanticIDGroundingIndex

    payload_cols = [
        c
        for c in [
            "placekey_id",
            "placekey",
            "top_category",
            "naics_code",
            "latitude",
            "longitude",
        ]
        if c in metadata_frame.columns
    ]
    id_column = (
        "placekey_id" if "placekey_id" in codes_df.columns else "placekey"
    )
    grounding_frame = codes_df.join(
        metadata_frame.select(payload_cols).unique(id_column, keep="first"),
        on=id_column,
        how="left",
        maintain_order="left",
    )
    grounding_index = SemanticIDGroundingIndex.from_frame(
        grounding_frame,
        key_columns=code_cols,
        id_column=id_column,
        payload_columns=payload_cols,
    )
    grounding_index.save(grounding_index_path)
    return grounding_index
