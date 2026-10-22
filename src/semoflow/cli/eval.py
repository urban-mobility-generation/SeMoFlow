from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Literal

import polars as pl
import typer

from semoflow.cli.common import (
    ConfigOpt,
    ForceOpt,
    LogFileOpt,
    LogLevelOpt,
    SetArgsOpt,
    bad_parameter_on,
    echo_json,
    resolve_cli_config,
    validate_device,
)
from semoflow.logging.primitives import configure_logging

app = typer.Typer(
    help="Evaluate planner samples, compositions, and tokenizers.",
    no_args_is_help=True,
    add_completion=False,
)
logger = logging.getLogger(__name__)


@app.command(
    "sample-plans",
    help="Sample occurrence plans from a trained planner checkpoint.",
)
def sample_plans_cmd(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint", help="Planner checkpoint (best.pt/last.pt)."
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Sampled-sequence artifact path (.pt)."),
    ],
    conditions: Annotated[
        Path | None,
        typer.Option(
            "--conditions",
            help="Optional exact lengths/contexts artifact (.pt).",
        ),
    ] = None,
    num_samples: Annotated[int, typer.Option("--num-samples")] = 1,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Maximum number of sequences sampled per generation batch.",
        ),
    ] = 1024,
    temperature: Annotated[float, typer.Option("--temperature")] = 1.0,
    length_temperature: Annotated[
        float | None, typer.Option("--length-temperature")
    ] = None,
    gap_temperature: Annotated[
        float | None, typer.Option("--gap-temperature")
    ] = None,
    prefix_temperature: Annotated[
        float | None, typer.Option("--prefix-temperature")
    ] = None,
    route_temperature: Annotated[
        float | None,
        typer.Option(
            "--route-temperature",
            help="Explore/Retrieve gate temperature; defaults to --temperature.",
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    validate_device(device)
    for name, value in (
        ("--temperature", temperature),
        ("--length-temperature", length_temperature),
        ("--gap-temperature", gap_temperature),
        ("--prefix-temperature", prefix_temperature),
        ("--route-temperature", route_temperature),
    ):
        if value is not None and value <= 0:
            raise typer.BadParameter(f"{name} must be positive")

    from semoflow.planner.sampling_run import (
        sample_occurrence_planner,
    )

    with bad_parameter_on(ValueError, FileExistsError):
        result = sample_occurrence_planner(
            checkpoint_path=checkpoint,
            output_path=output,
            num_samples=num_samples,
            batch_size=batch_size,
            conditions_path=conditions,
            device=device,
            seed=seed,
            temperature=temperature,
            length_temperature=length_temperature,
            gap_temperature=gap_temperature,
            prefix_temperature=prefix_temperature,
            route_temperature=route_temperature,
            force=force,
        )
    echo_json(
        {
            "output_path": result.output_path,
            "num_samples": result.num_samples,
            "batch_size": result.batch_size,
        }
    )


@app.command(
    "planner-samples",
    help=(
        "Validate and summarize a saved occurrence-planner sample artifact."
    ),
)
def eval_planner_samples_cmd(
    samples: Annotated[
        Path,
        typer.Option(
            "--samples",
            help="Saved occurrence_planner_samples artifact to validate.",
        ),
    ],
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="Occurrence planner checkpoint used to generate the samples.",
        ),
    ],
    reference: Annotated[
        Path,
        typer.Option(
            "--reference",
            help="Hash-matched occurrence_planner_dataset reference artifact.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            help="JSON validation report path.",
        ),
    ],
    reference_split: Annotated[
        str,
        typer.Option(
            "--reference-split",
            help="Dataset split used for distribution comparisons.",
        ),
    ] = "validation",
    reference_limit: Annotated[
        int | None,
        typer.Option(
            "--reference-limit",
            help=(
                "Use a seeded N-row reference panel; default uses the full "
                "split."
            ),
        ),
    ] = None,
    evaluation_seed: Annotated[
        int,
        typer.Option(
            "--evaluation-seed",
            help="Seed for reference-panel and real-floor sampling.",
        ),
    ] = 0,
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=(
                "Single compute device for router diagnostics, for example "
                "cpu, cuda, or cuda:3. Other metrics remain on CPU."
            ),
        ),
    ] = "cpu",
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            min=1,
            help="Maximum reference rows per router-diagnostic forward pass.",
        ),
    ] = 1024,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.metrics.occurrence_planner_samples import (
        OccurrencePlannerSamplesEvaluationConfig,
        evaluate_occurrence_planner_samples,
    )

    with bad_parameter_on(OSError, ValueError):
        report = evaluate_occurrence_planner_samples(
            OccurrencePlannerSamplesEvaluationConfig(
                samples=samples,
                checkpoint=checkpoint,
                reference=reference,
                output=output,
                reference_split=reference_split,
                reference_limit=reference_limit,
                evaluation_seed=evaluation_seed,
                device=device,
                batch_size=batch_size,
            )
        )
    echo_json(
        {
            "output": output,
            "sample_count": report["sample_count"],
            "reference_count": report["reference_count"],
            "legal_prefix_rate": report["legal_prefix_rate"],
            "event_time_monotonic_rate": report["event_time_monotonic_rate"],
            "comparisons": report["population_metrics"]["comparisons"],
            "real_vs_real_floor": report["real_vs_real_floor"]["comparisons"],
        }
    )


@app.command(
    "compose",
    help=("Compose planner samples with the latent-flow suffix realizer."),
)
def eval_compose_cmd(
    realizer_checkpoint: Annotated[
        Path,
        typer.Option(
            "--realizer-checkpoint",
            help="Trained PrefixTimeSuffixLatentFlowRealizer checkpoint.",
        ),
    ],
    planner_samples: Annotated[
        Path,
        typer.Option(
            "--planner-samples",
            help=(
                "Occurrence planner samples or a sampled_prefix_time_plan "
                "artifact."
            ),
        ),
    ],
    bank: Annotated[
        Path,
        typer.Option(
            "--bank",
            help="prefix_time_suffix_bank.pt containing the validation split.",
        ),
    ],
    codebooks: Annotated[
        Path,
        typer.Option("--codebooks", help="Tokenizer codebook bundle."),
    ],
    lookup: Annotated[
        Path,
        typer.Option("--lookup", help="Semantic-ID placekey lookup parquet."),
    ],
    grounding_index: Annotated[
        Path,
        typer.Option("--grounding-index", help="Semantic-ID grounding index."),
    ],
    places: Annotated[
        Path,
        typer.Option("--places", help="Place metadata parquet."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Composition evaluation JSON path."),
    ],
    planner_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--planner-checkpoint",
            help=(
                "Optional occurrence planner checkpoint. When supplied, the "
                "saved sample hash and planner dimensions are verified."
            ),
        ),
    ] = None,
    max_samples: Annotated[
        int,
        typer.Option("--max-samples", help="Maximum trajectories per arm."),
    ] = 10000,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Maximum latent-flow sampling batch size.",
        ),
    ] = 64,
    device: Annotated[
        str,
        typer.Option("--device", help="Torch device, for example cuda:6."),
    ] = "cuda:0",
    num_steps: Annotated[
        int,
        typer.Option(
            "--num-steps", help="Number of latent-flow solver steps."
        ),
    ] = 32,
    solver: Annotated[
        Literal["euler", "heun"],
        typer.Option("--solver", help="Latent-flow ODE solver."),
    ] = "heun",
    reference_split: Annotated[
        Literal["val", "test"],
        typer.Option(
            "--reference-split", help="Held-out reference bank partition."
        ),
    ] = "val",
    candidate_selection: Annotated[
        Literal[
            "first", "train_prior_argmax", "train_prior_sample", "a4_support"
        ],
        typer.Option(
            "--candidate-selection", help="Generated-side grounding policy."
        ),
    ] = "first",
    candidate_prior: Annotated[
        Path | None,
        typer.Option("--candidate-prior", help="Train-split candidate prior."),
    ] = None,
    catalog_temperature: Annotated[
        float | None,
        typer.Option(
            "--catalog-temperature",
            help="Override the calibrated catalog decoder temperature.",
        ),
    ] = None,
    catalog_prior_source: Annotated[
        Literal["target_train", "uniform"],
        typer.Option("--catalog-prior-source", help="Catalog prior source."),
    ] = "target_train",
    generation_catalog_scope: Annotated[
        Literal["full_city", "train_observed"],
        typer.Option(
            "--generation-catalog-scope",
            help="Generated-side candidate scope.",
        ),
    ] = "full_city",
    flow_trace_rows: Annotated[
        int,
        typer.Option(
            "--flow-trace-rows", help="Persist ODE traces for N rows."
        ),
    ] = 0,
    flow_path_output: Annotated[
        Path | None,
        typer.Option("--flow-path-output", help="Optional ODE trace path."),
    ] = None,
    brief_metrics: Annotated[
        bool,
        typer.Option(
            "--brief-metrics",
            help=(
                "Compute only the six comparison metrics and POI coverage; "
                "skip detailed diagnostics and bootstrap intervals."
            ),
        ),
    ] = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.metrics.planner_realizer_composition import (
        evaluate_planner_realizer,
    )

    with bad_parameter_on(OSError, RuntimeError, ValueError):
        report = evaluate_planner_realizer(
            realizer_checkpoint=realizer_checkpoint,
            planner_samples=planner_samples,
            planner_checkpoint=planner_checkpoint,
            bank=bank,
            codebooks=codebooks,
            lookup=lookup,
            grounding_index=grounding_index,
            places=places,
            output=output,
            max_samples=max_samples,
            batch_size=batch_size,
            device=device,
            num_steps=num_steps,
            solver=solver,
            reference_split=reference_split,
            candidate_selection=candidate_selection,
            candidate_prior=candidate_prior,
            catalog_temperature=catalog_temperature,
            catalog_prior_source=catalog_prior_source,
            generation_catalog_scope=generation_catalog_scope,
            flow_trace_rows=flow_trace_rows,
            flow_path_output=flow_path_output,
            brief_metrics=brief_metrics,
        )

    echo_json(
        {"output": output, "rows": report.get("rows"), **report["summary"]}
    )


@app.command(
    "tokenizer", help="Evaluate a published Semantic-ID tokenizer run."
)
def eval_tokenizer_cmd(
    config: ConfigOpt = None,
    processed_visits_path: Annotated[
        Path | None,
        typer.Option(
            "--stays-path",
            help="Path to processed_visits.parquet for trajectory-level metrics.",
        ),
    ] = None,
    overrides: SetArgsOpt = None,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.config.artifacts import TokenizerArtifacts
    from semoflow.preprocess.places_index import (
        load_places_index,
    )

    cfg = resolve_cli_config(
        config,
        overrides,
        log_level=log_level,
        log_file=log_file,
    )
    tcfg = cfg.tokenizer_config
    artifacts = TokenizerArtifacts.from_run_dir(tcfg.run_dir())
    places_index = load_places_index(tcfg.places_index_output_path())

    stays = None
    resolved_processed_visits_path = (
        processed_visits_path or tcfg.processed_visits_output_path()
    )
    if resolved_processed_visits_path.exists():
        stays = pl.read_parquet(resolved_processed_visits_path)
        logger.info(
            "loaded stays for trajectory metrics: %s (%d rows)",
            resolved_processed_visits_path,
            len(stays),
        )

    from semoflow.training.tokenizer_publish import (
        run_semantic_id_eval,
    )

    run_semantic_id_eval(
        cfg=tcfg,
        artifacts=artifacts,
        places_index=places_index,
        stays=stays,
    )


@app.command(
    "tokenizer-hierarchy",
    help="Audit per-depth SID prefix branching, purity, and coherence.",
)
def eval_tokenizer_hierarchy_cmd(
    grounding_index: Annotated[
        Path,
        typer.Option(
            "--grounding-index",
            help="Path to grounding_index.pkl for the tokenizer under audit.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination JSON artifact."),
    ],
    label: Annotated[
        str | None,
        typer.Option(
            "--label",
            help="Optional tokenizer label recorded in the artifact.",
        ),
    ] = None,
    stays: Annotated[
        Path | None,
        typer.Option(
            "--stays",
            help="Optional processed-visits parquet enabling the "
            "co-visitation coherence section.",
        ),
    ] = None,
    covisit_split: Annotated[
        str,
        typer.Option(
            "--covisit-split",
            help="Split used for co-visitation coherence. MUST be held "
            "out: an encoder trained on train co-visitation scored "
            "against train co-visitation is close to tautological.",
        ),
    ] = "val",
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    configure_logging(level=log_level, log_file=log_file)
    from semoflow.metrics.semantic_id_hierarchy import (
        audit_sid_hierarchy,
    )

    report = audit_sid_hierarchy(
        grounding_index,
        stays_path=stays,
        covisit_split=covisit_split,
        label=label,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    logger.info(
        "POIs=%d  distinct full SIDs=%d  joint=%.2f bits  injectivity=%.4f",
        report["n_pois"],
        report["distinct_full_sids"],
        report["joint_sid_entropy_bits_upper_bound"],
        report["sid_injectivity"],
    )
    for row in report["prefix_depths"]:
        branch = row["branching_factor"]
        logger.info(
            "depth=%d  distinct=%d  branch=%s  POIs/prefix=%.2f  "
            "cond_bits=%.2f  cat_purity=%s  naics2_purity=%s  geo_km=%s",
            row["depth"],
            row["distinct_prefixes"],
            f"{branch:.2f}x" if branch else "---",
            row["mean_pois_per_prefix"],
            row["conditional_bits_upper_bound"],
            row.get("category_purity"),
            row.get("naics2_purity"),
            row.get("mean_geo_radius_km"),
        )
    logger.info("wrote %s", output)


@app.command(
    "encoder-embeddings",
    help="Evaluate row-aligned encoder embedding artifacts.",
)
def eval_encoder_embeddings_cmd(
    embeddings: Annotated[
        Path,
        typer.Option(
            "--embeddings",
            help="Path to .npy or .pt row-aligned embedding artifact.",
        ),
    ],
    index: Annotated[
        Path | None,
        typer.Option(
            "--index",
            help="Optional row-aligned POI index parquet with top_category.",
        ),
    ] = None,
    paired_embeddings: Annotated[
        Path | None,
        typer.Option(
            "--paired-embeddings",
            help="Optional paired embedding artifact for text/graph alignment.",
        ),
    ] = None,
    paired_kind: Annotated[
        str | None,
        typer.Option(
            "--paired-kind",
            help="Kind of paired embeddings, usually text or graph.",
        ),
    ] = None,
    processed_visits_path: Annotated[
        Path | None,
        typer.Option(
            "--stays-path",
            help="Optional processed visits parquet for behavior metrics.",
        ),
    ] = None,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir", help="Directory for embedding eval outputs."
        ),
    ] = Path("outputs/encoder/embedding_eval"),
    kind: Annotated[
        Literal["text", "graph", "cf"],
        typer.Option("--kind", help="Embedding kind."),
    ] = "text",
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    configure_logging(level=log_level, log_file=log_file)

    from semoflow.metrics.encoder_embeddings import (
        write_embedding_eval,
    )

    write_embedding_eval(
        embeddings,
        out_dir=out_dir,
        index_path=index,
        paired_embeddings=paired_embeddings,
        paired_kind=paired_kind,
        processed_visits_path=processed_visits_path,
        kind=kind,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
