from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, cast

import typer

from semoflow.cli.common import (
    ConfigOpt,
    ForceOpt,
    LogLevelOpt,
    SetArgsOpt,
    bad_parameter_on,
    echo_json,
    resolve_cli_config,
)
from semoflow.logging.primitives import configure_logging
from semoflow.utils.serialization import write_json

app = typer.Typer(
    help="Derive and assemble identity-keyed SeMoFlow artifacts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    "build",
    help=(
        "Build a stage's training data from its config: the weekly segments "
        "and planner dataset for a planner config, or the bank for a "
        "realizer config."
    ),
)
def build_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    from semoflow.planner.dataset_build import build_planner_data
    from semoflow.realizer.bank import build_realizer_data

    cfg = resolve_cli_config(config, set_args, log_level=log_level)
    with bad_parameter_on(FileExistsError, OSError, TypeError, ValueError):
        if cfg.planner is not None and cfg.planner.data is not None:
            result = build_planner_data(
                cfg.planner.data,
                dataset_path=cfg.planner.dataset_path,
                force=force,
            )
        elif cfg.realizer is not None and cfg.realizer.data is not None:
            result = build_realizer_data(cfg.realizer)
        else:
            raise ValueError(
                "--config must define planner.data or realizer.data"
            )
    echo_json(result)


@app.command(
    "build-weekly-segments",
    help="Build the weekly segment cache read by the planner and realizer.",
)
def build_weekly_segments_cmd(
    stays: Annotated[
        Path,
        typer.Option(
            "--stays",
            help="SID-augmented visits (augmented_processed_visits.parquet).",
        ),
    ],
    output: Annotated[Path, typer.Option("--output")],
    min_seq_len: Annotated[int, typer.Option("--min-seq-len")] = 5,
    min_active_days: Annotated[int, typer.Option("--min-active-days")] = 5,
    max_seq_len: Annotated[int, typer.Option("--max-seq-len")] = 128,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.preprocess.weekly_segments import build_weekly_segment_cache

    with bad_parameter_on(FileExistsError, OSError, ValueError):
        result = build_weekly_segment_cache(
            stays_path=stays,
            output=output,
            min_seq_len=min_seq_len,
            min_active_days=min_active_days,
            max_seq_len=max_seq_len,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-realizer-bank",
    help=(
        "Build an input-only time-aware SID bank for the latent-flow realizer."
    ),
)
def build_realizer_bank_cmd(
    lookup: Annotated[Path, typer.Option("--lookup")],
    segments: Annotated[Path, typer.Option("--segments")],
    output: Annotated[Path, typer.Option("--output")],
    max_events: Annotated[int, typer.Option("--max-events")] = 32,
    min_length: Annotated[int, typer.Option("--min-length")] = 5,
    max_train_segments: Annotated[
        int, typer.Option("--max-train-segments")
    ] = 0,
    max_val_segments: Annotated[int, typer.Option("--max-val-segments")] = 0,
    max_test_segments: Annotated[int, typer.Option("--max-test-segments")] = 0,
    allow_empty_splits: Annotated[
        bool,
        typer.Option(
            "--allow-empty-splits/--require-nonempty-splits",
            help=(
                "Permit empty train/test windows for reference-only banks; "
                "the validation split must remain non-empty."
            ),
        ),
    ] = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.realizer.bank import build_bank

    with bad_parameter_on(OSError, TypeError, ValueError):
        result = build_bank(
            lookup_path=lookup,
            segments_path=segments,
            output_path=output,
            max_events=max_events,
            min_length=min_length,
            max_train_segments=max_train_segments,
            max_val_segments=max_val_segments,
            max_test_segments=max_test_segments,
            allow_empty_splits=allow_empty_splits,
        )
    echo_json(result)


@app.command(
    "build-grounding-candidate-prior",
    help="Fit a train-only POI occurrence prior for catalog grounding.",
)
def build_grounding_candidate_prior_cmd(
    bank: Annotated[Path, typer.Option("--bank")],
    output: Annotated[Path, typer.Option("--output")],
    split: Annotated[str, typer.Option("--split")] = "train",
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.metrics.grounding_candidate_prior import build_prior

    with bad_parameter_on(OSError, TypeError, ValueError):
        prior = build_prior(bank, split=split)
        write_json(output, prior)
    summary = {key: value for key, value in prior.items() if key != "counts"}
    echo_json({"output": str(output), **summary})


@app.command(
    "extract-city",
    help=(
        "Write one city's segments, SID lookup, and grounding index from "
        "joint multi-city artifacts, for per-city banks, priors, and "
        "composition."
    ),
)
def extract_city_cmd(
    city_id: Annotated[int, typer.Option("--city-id")],
    segments: Annotated[Path, typer.Option("--segments")],
    lookup: Annotated[Path, typer.Option("--lookup")],
    grounding_index: Annotated[Path, typer.Option("--grounding-index")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.metrics.city_inputs import extract_city_inputs

    with bad_parameter_on(FileExistsError, OSError, TypeError, ValueError):
        result = extract_city_inputs(
            city_id,
            segments=segments,
            lookup=lookup,
            grounding_index=grounding_index,
            output_dir=output_dir,
            force=force,
        )
    echo_json(result)


def _parse_sources(values: list[str]) -> dict[int, Path]:
    sources: dict[int, Path] = {}
    for value in values:
        try:
            raw_city_id, raw_path = value.split("=", 1)
            city_id = int(raw_city_id)
        except ValueError as exc:
            raise typer.BadParameter(
                f"source must be CITY_ID=PATH, got {value!r}"
            ) from exc
        if city_id in sources:
            raise typer.BadParameter(f"city ID {city_id} was supplied twice")
        sources[city_id] = Path(raw_path)
    return sources


@app.command(
    "subset-cities",
    help="Derive a remapped city subset from joint preprocessing artifacts.",
)
def subset_cities_cmd(
    source_dir: Annotated[Path, typer.Option("--source-dir")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    city_ids: Annotated[list[int], typer.Option("--city-id")],
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.preprocess.city_subset import derive_city_subset

    result = derive_city_subset(source_dir, output_dir, city_ids)
    echo_json(result.to_dict())


@app.command(
    "assemble-text",
    help="Assemble per-city text embeddings in a joint POI namespace.",
)
def assemble_text_cmd(
    places_index: Annotated[Path, typer.Option("--places-index")],
    source: Annotated[
        list[str],
        typer.Option(
            "--source",
            help="Per-city text artifact as CITY_ID=PATH (repeatable).",
        ),
    ],
    output: Annotated[Path, typer.Option("--output")],
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.encoder.text.assembly import assemble_text_embeddings

    result = assemble_text_embeddings(
        places_index, _parse_sources(source), output
    )
    echo_json(result.to_dict())


@app.command(
    "size-tokenizer",
    help="Estimate a city's tokenizer codebook size and residual depth.",
)
def size_tokenizer_cmd(
    places: Annotated[Path, typer.Option("--places")],
    text_dir: Annotated[Path, typer.Option("--text-dir")],
    graph_dir: Annotated[Path, typer.Option("--graph-dir")],
    codebook_sizes: Annotated[
        list[int] | None, typer.Option("--codebook-size")
    ] = None,
    max_levels: Annotated[int, typer.Option("--max-levels")] = 8,
    pca_text_dim: Annotated[int, typer.Option("--pca-text-dim")] = 16,
    text_graph_scale: Annotated[
        float, typer.Option("--text-graph-scale")
    ] = 4.0,
    device: Annotated[str, typer.Option("--device")] = "cuda:0",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.tokenization.sizing import size_tokenizer

    report = size_tokenizer(
        places=places,
        text_dir=text_dir,
        graph_dir=graph_dir,
        codebook_sizes=codebook_sizes or [32, 48, 64],
        max_levels=max_levels,
        pca_text_dim=pca_text_dim,
        text_graph_scale=text_graph_scale,
        device=device,
    )
    payload = json.dumps(asdict(report), indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload, nl=False)


@app.command(
    "assemble-composed-tokenizer",
    help=(
        "Assemble a per-city tokenizer directory over a composed "
        "shared+local SID."
    ),
)
def assemble_composed_tokenizer_cmd(
    composed_lookup: Annotated[Path, typer.Option("--composed-lookup")],
    city_id: Annotated[int, typer.Option("--city-id")],
    local_dir: Annotated[
        Path,
        typer.Option(
            "--local-dir",
            help="City's local tower run directory; supplies its "
            "augmented visits and local codebooks.",
        ),
    ],
    local_places: Annotated[
        Path,
        typer.Option(
            "--local-places",
            help="places_index.parquet from the local tower's "
            "preprocessing run; the visit namespace is bridged through "
            "placekey, never placekey_id.",
        ),
    ],
    shared_codebooks: Annotated[Path, typer.Option("--shared-codebooks")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.tokenization.composed_dir import (
        assemble_composed_tokenizer_dir,
    )

    result = assemble_composed_tokenizer_dir(
        composed_lookup=composed_lookup,
        city_id=city_id,
        local_dir=local_dir,
        local_places=local_places,
        shared_codebooks=shared_codebooks,
        output_dir=output_dir,
    )
    echo_json(result.to_dict())


@app.command(
    "category-relative-geo-assignment",
    help="Derive a category + city-relative geography control tokenizer.",
)
def category_relative_geo_assignment_cmd(
    source_dir: Annotated[
        Path,
        typer.Option(
            "--source-dir",
            help="Fitted tokenizer run directory the control is matched to.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    places_index: Annotated[Path, typer.Option("--places-index")],
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="Tokenizer checkpoint supplying the fit-time feature space "
            "the suffix levels are clustered in.",
        ),
    ],
    seed: Annotated[int, typer.Option("--seed")],
    branching: Annotated[
        int | None,
        typer.Option(
            "--branching",
            help="Quantile strips per axis per geography level; the level "
            "uses branching**2 codes. Defaults to isqrt(codebook_size).",
        ),
    ] = None,
    text_embeddings: Annotated[
        Path | None, typer.Option("--text-embeddings")
    ] = None,
    graphsage_artifact_dir: Annotated[
        Path | None, typer.Option("--graphsage-artifact-dir")
    ] = None,
    city_id: Annotated[list[int] | None, typer.Option("--city-id")] = None,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.tokenization.category_relative_geo_assignment import (
        write_category_relative_geo_tokenizer_dir,
    )

    result = write_category_relative_geo_tokenizer_dir(
        source_dir=source_dir,
        output_dir=output_dir,
        places_index=places_index,
        checkpoint=checkpoint,
        text_embeddings=text_embeddings,
        graphsage_artifact_dir=graphsage_artifact_dir,
        city_ids=city_id or None,
        seed=seed,
        branching=branching,
    )
    echo_json(result.to_dict())


@app.command(
    "encode-frozen-tokenizer",
    help=(
        "Encode a POI set through an already-fitted tokenizer without "
        "refitting."
    ),
)
def encode_frozen_tokenizer_cmd(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    places_index: Annotated[Path, typer.Option("--places-index")],
    output: Annotated[Path, typer.Option("--output")],
    city_id: Annotated[
        list[int] | None,
        typer.Option(
            "--city-id",
            help="Repeatable. Restrict to these cities; default is every city.",
        ),
    ] = None,
    text_embeddings: Annotated[
        Path | None,
        typer.Option(
            "--text-embeddings",
            help="Required when the checkpoint's input source reads embeddings.",
        ),
    ] = None,
    graphsage_artifact_dir: Annotated[
        Path | None,
        typer.Option(
            "--graphsage-artifact-dir",
            help="Target GraphSAGE artifact directory for graph/concat "
            "inputs; defaults to the checkpoint config.",
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.tokenization.frozen_encode import (
        encode_with_frozen_tokenizer,
    )

    result = encode_with_frozen_tokenizer(
        checkpoint=checkpoint,
        places_index=places_index,
        output=output,
        city_ids=city_id or None,
        text_embeddings=text_embeddings,
        graphsage_artifact_dir=graphsage_artifact_dir,
        device=device,
    )
    echo_json(result.to_dict())


@app.command(
    "apply-cross-city-projection",
    help=(
        "Apply a frozen cross-city projection to a city it was never "
        "fitted on."
    ),
)
def apply_cross_city_projection_cmd(
    projection: Annotated[Path, typer.Option("--projection")],
    embeddings: Annotated[Path, typer.Option("--embeddings")],
    embeddings_index: Annotated[Path, typer.Option("--embeddings-index")],
    places_index: Annotated[Path, typer.Option("--places-index")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    source_template: Annotated[
        str,
        typer.Option(
            "--source-template",
            help="Template the source embeddings were encoded with "
            "(address_stripped or full). Must match the projection's "
            "training template.",
        ),
    ],
    city_id: Annotated[list[int] | None, typer.Option("--city-id")] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    if source_template not in ("address_stripped", "full"):
        raise typer.BadParameter(
            "--source-template must be address_stripped or full"
        )
    from semoflow.tokenization.apply_cross_city_projection import (
        apply_cross_city_projection,
    )

    result = apply_cross_city_projection(
        projection=projection,
        embeddings=embeddings,
        embeddings_index=embeddings_index,
        places_index=places_index,
        output_dir=output_dir,
        source_template=source_template,
        city_ids=city_id or None,
        device=device,
    )
    echo_json(result.to_dict())


@app.command(
    "build-occurrence-planner-dataset",
    help="Build a strict modular-planner dataset from weekly segments.",
)
def build_occurrence_planner_dataset_cmd(
    segments: Annotated[Path, typer.Option("--segments")],
    lookup: Annotated[Path, typer.Option("--lookup")],
    output: Annotated[Path, typer.Option("--output")],
    codebook_size: Annotated[int, typer.Option("--codebook-size")],
    codebooks: Annotated[
        Path | None,
        typer.Option(
            "--codebooks",
            help=(
                "tokenizer_codebooks.pt whose first --prefix-depth levels "
                "become the frozen latent for --tokenizer-latent training."
            ),
        ),
    ] = None,
    places: Annotated[
        list[Path] | None,
        typer.Option(
            "--places",
            help="Repeatable places tables supplying placekey/geohash_5 for origin context.",
        ),
    ] = None,
    prefix_depth: Annotated[int, typer.Option("--prefix-depth")] = 3,
    min_occurrences: Annotated[int, typer.Option("--min-occurrences")] = 1,
    max_occurrences: Annotated[int, typer.Option("--max-occurrences")] = 30,
    overflow_policy: Annotated[
        str,
        typer.Option(
            "--overflow-policy",
            help="How to handle rows longer than --max-occurrences: "
            "'truncate', 'drop', or 'error'.",
        ),
    ] = "truncate",
    context_mode: Annotated[
        str,
        typer.Option(
            "--context-mode",
            help=(
                "'none', typed 'city', or generation-time "
                "'city_calendar_origin'."
            ),
        ),
    ] = "none",
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.planner.dataset_build import build_occurrence_planner_dataset

    with bad_parameter_on(FileExistsError, TypeError, ValueError):
        result = build_occurrence_planner_dataset(
            segments_path=segments,
            lookup_path=lookup,
            places_paths=places or None,
            output_path=output,
            codebook_size=codebook_size,
            codebooks_path=codebooks,
            prefix_depth=prefix_depth,
            min_occurrences=min_occurrences,
            max_occurrences=max_occurrences,
            overflow_policy=overflow_policy,
            context_mode=context_mode,
            force=force,
        )
    echo_json(result.to_dict())


@app.command(
    "build-occurrence-planner-conditions",
    help="Build checkpoint-matched natural generation conditions.",
)
def build_occurrence_planner_conditions_cmd(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            help=(
                "Parquet rows with city_id and optional length; "
                "city_calendar_origin checkpoints additionally require "
                "week_start and origin_geohash_5."
            ),
        ),
    ],
    output: Annotated[Path, typer.Option("--output")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    import polars as pl

    from semoflow.planner.conditions_build import (
        build_occurrence_planner_conditions,
    )

    with bad_parameter_on(FileExistsError, OSError, TypeError, ValueError):
        result = build_occurrence_planner_conditions(
            checkpoint_path=checkpoint,
            rows=pl.read_parquet(input_path),
            output_path=output,
            force=force,
        )
    echo_json(result.to_dict())


_ARTIFACT_ERRORS = (FileExistsError, OSError, RuntimeError, ValueError)


@app.command(
    "build-budget-ladder",
    help="Select nested, device-disjoint target-train budget rungs.",
)
def build_budget_ladder_cmd(
    city: Annotated[str, typer.Option("--city")],
    visits: Annotated[
        Path, typer.Option("--visits", help="Target processed_visits parquet.")
    ],
    segments: Annotated[
        Path, typer.Option("--segments", help="Target weekly segment cache.")
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    fraction: Annotated[
        list[float] | None,
        typer.Option(
            "--fraction",
            help=(
                "Repeatable budget fraction in (0, 1]; default "
                "0.01 0.05 0.1 0.2 0.5 1.0. The 0% rung has no panel."
            ),
        ),
    ] = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.budget_ladder import (
        DEFAULT_FRACTIONS,
        build_budget_ladder,
    )

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_budget_ladder(
            city=city,
            visits=visits,
            segments=segments,
            output_dir=output_dir,
            fractions=fraction or DEFAULT_FRACTIONS,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-budget-rung-visits",
    help="Write one rung's inner-role visits for the rung's visit graph.",
)
def build_budget_rung_visits_cmd(
    visits: Annotated[Path, typer.Option("--visits")],
    panel: Annotated[
        Path, typer.Option("--panel", help="A rung's devices.parquet.")
    ],
    output: Annotated[Path, typer.Option("--output")],
    inner_role: Annotated[str, typer.Option("--inner-role")] = "train",
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.budget_ladder import build_rung_visits

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_rung_visits(
            visits=visits,
            panel=panel,
            output=output,
            inner_role=inner_role,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-budget-rung-augmented-visits",
    help="Write one rung's union-numbered, SID-coded visits for SID banks.",
)
def build_budget_rung_augmented_visits_cmd(
    visits: Annotated[Path, typer.Option("--visits")],
    panel: Annotated[Path, typer.Option("--panel")],
    places_index: Annotated[
        Path,
        typer.Option("--places-index", help="Target places index (bridge)."),
    ],
    union_lookup: Annotated[Path, typer.Option("--union-lookup")],
    union_manifest: Annotated[Path, typer.Option("--union-manifest")],
    output: Annotated[Path, typer.Option("--output")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.budget_ladder import build_rung_augmented_visits

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_rung_augmented_visits(
            visits=visits,
            panel=panel,
            places_index=places_index,
            union_lookup=union_lookup,
            union_manifest=union_manifest,
            output=output,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-holdout-reference-panel",
    help="Select the campaign's fixed, seed-determined held-out reference.",
)
def build_holdout_reference_panel_cmd(
    segments: Annotated[Path, typer.Option("--segments")],
    output: Annotated[Path, typer.Option("--output")],
    split: Annotated[
        str, typer.Option("--split", help="Outer held-out split.")
    ] = "val",
    size: Annotated[int, typer.Option("--size")] = 20_000,
    min_length: Annotated[int, typer.Option("--min-length")] = 5,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.budget_ladder import build_holdout_reference_panel

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_holdout_reference_panel(
            segments=segments,
            output=output,
            split=split,
            size=size,
            min_length=min_length,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-budget-rung-segments",
    help=(
        "Write one rung's union-numbered weekly segments plus the reference."
    ),
)
def build_budget_rung_segments_cmd(
    segments: Annotated[Path, typer.Option("--segments")],
    panel: Annotated[Path, typer.Option("--panel")],
    union_lookup: Annotated[Path, typer.Option("--union-lookup")],
    union_manifest: Annotated[Path, typer.Option("--union-manifest")],
    holdout_panel: Annotated[Path, typer.Option("--holdout-panel")],
    output: Annotated[Path, typer.Option("--output")],
    layout: Annotated[
        str,
        typer.Option(
            "--layout",
            help=(
                "'train': inner validation in val, reference in test. "
                "'eval': reference in val, inner validation in test."
            ),
        ),
    ] = "train",
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.budget_ladder import (
        RungSegmentLayout,
        build_rung_segments,
    )

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_rung_segments(
            segments=segments,
            panel=panel,
            union_lookup=union_lookup,
            union_manifest=union_manifest,
            holdout_panel=holdout_panel,
            output=output,
            layout=cast(RungSegmentLayout, layout),
            force=force,
        )
    echo_json(result)


@app.command(
    "build-union-lookup",
    help="Union a source multi-city lookup with a renumbered target lookup.",
)
def build_union_lookup_cmd(
    source_lookup: Annotated[Path, typer.Option("--source-lookup")],
    target_lookup: Annotated[Path, typer.Option("--target-lookup")],
    output: Annotated[Path, typer.Option("--output")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.union_catalog import build_union_lookup

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_union_lookup(
            source_lookup=source_lookup,
            target_lookup=target_lookup,
            output=output,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-union-places-index",
    help="Union places indexes in the union lookup's numbering.",
)
def build_union_places_index_cmd(
    source_places: Annotated[Path, typer.Option("--source-places")],
    target_places: Annotated[Path, typer.Option("--target-places")],
    union_lookup: Annotated[Path, typer.Option("--union-lookup")],
    union_manifest: Annotated[Path, typer.Option("--union-manifest")],
    output: Annotated[Path, typer.Option("--output")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.union_catalog import build_union_places_index

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_union_places_index(
            source_places=source_places,
            target_places=target_places,
            union_lookup=union_lookup,
            union_manifest=union_manifest,
            output=output,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-union-grounding-index",
    help="Build a grounding index spanning every union catalog row.",
)
def build_union_grounding_index_cmd(
    union_lookup: Annotated[Path, typer.Option("--union-lookup")],
    places: Annotated[
        list[Path],
        typer.Option(
            "--places",
            help="Repeatable places tables supplying placekey/top_category.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output")],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.union_catalog import build_union_grounding_index

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = build_union_grounding_index(
            union_lookup=union_lookup,
            places=places,
            output=output,
            force=force,
        )
    echo_json(result)


@app.command(
    "derive-target-city-tokenizer",
    help=(
        "Renumber a target city and add its catalog centroid to a tokenizer."
    ),
)
def derive_target_city_tokenizer_cmd(
    tokenizer: Annotated[
        Path,
        typer.Option(
            "--tokenizer", help="Frozen tokenizer_model.pt of the source fit."
        ),
    ],
    places_index: Annotated[
        Path, typer.Option("--places-index", help="Target places index.")
    ],
    target_city_id: Annotated[
        int,
        typer.Option(
            "--target-city-id", help="Unused city id to assign the target."
        ),
    ],
    output_tokenizer: Annotated[Path, typer.Option("--output-tokenizer")],
    output_places_index: Annotated[
        Path, typer.Option("--output-places-index")
    ],
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.tokenization.target_city import derive_target_city_tokenizer

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = derive_target_city_tokenizer(
            tokenizer=tokenizer,
            places_index=places_index,
            target_city_id=target_city_id,
            output_tokenizer=output_tokenizer,
            output_places_index=output_places_index,
            force=force,
        )
    echo_json(result)


@app.command(
    "encode-catalog-only-graph",
    help=(
        "Encode 0%-rung POI embeddings from an untrained catalog-only graph."
    ),
)
def encode_catalog_only_graph_cmd(
    places_index: Annotated[Path, typer.Option("--places-index")],
    text_embeddings: Annotated[Path, typer.Option("--text-embeddings")],
    cbg_attributes: Annotated[
        Path,
        typer.Option(
            "--cbg-attributes",
            help="Required so the CBG input width matches the source.",
        ),
    ],
    warm_start_checkpoint: Annotated[
        Path, typer.Option("--warm-start-checkpoint")
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    hidden_dim: Annotated[int, typer.Option("--hidden-dim")] = 192,
    out_dim: Annotated[int, typer.Option("--out-dim")] = 128,
    num_layers: Annotated[int, typer.Option("--num-layers")] = 2,
    id_embedding_dim: Annotated[int, typer.Option("--id-embedding-dim")] = 64,
    dropout: Annotated[float, typer.Option("--dropout")] = 0.0,
    aggr: Annotated[str, typer.Option("--aggr")] = "mean",
    device: Annotated[str, typer.Option("--device")] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.cross_city.catalog_only_graph import (
        encode_catalog_only_graph,
    )

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = encode_catalog_only_graph(
            places_index=places_index,
            text_embeddings=text_embeddings,
            cbg_attributes=cbg_attributes,
            warm_start_checkpoint=warm_start_checkpoint,
            output_dir=output_dir,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            id_embedding_dim=id_embedding_dim,
            dropout=dropout,
            aggr=aggr,
            device=device,
            seed=seed,
            force=force,
        )
    echo_json(result)


@app.command(
    "build-untrained-planner-checkpoint",
    help=(
        "Save a warm-started planner checkpoint without training it on "
        "the target city.\n\n"
        "The architecture comes from --initialize-from. --batch-size, "
        "--learning-rate, and --weight-decay are recorded in the "
        "checkpoint but not applied."
    ),
)
def build_untrained_planner_checkpoint_cmd(
    dataset: Annotated[
        Path, typer.Option("--dataset", help="Target planner dataset.")
    ],
    initialize_from: Annotated[
        Path,
        typer.Option("--initialize-from", help="Source planner checkpoint."),
    ],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    batch_size: Annotated[int, typer.Option("--batch-size")] = 1024,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 5e-4,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 1e-4,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.planner.untrained_checkpoint import (
        publish_untrained_warm_start,
    )

    with bad_parameter_on(*_ARTIFACT_ERRORS):
        result = publish_untrained_warm_start(
            dataset_path=dataset,
            initialize_from=initialize_from,
            output_dir=output_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
            force=force,
        )
    echo_json(result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
