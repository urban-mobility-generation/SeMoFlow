from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

import typer

from semoflow.cli.common import (
    ConfigOpt,
    ForceOpt,
    LogFileOpt,
    LogLevelOpt,
    SetArgsOpt,
    bad_parameter_on,
    cli_run,
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
    "evaluate",
    help=(
        "For each evaluation city, sample plans, compose them into POI "
        "trajectories, and score them against the city's held-out weeks."
    ),
)
def evaluate_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.metrics.city_evaluation import evaluate_cities

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    settings, layout, cities = (
        cfg.evaluation,
        cfg.layout,
        cfg.evaluation_cities,
    )
    validate_device(settings.device, name="evaluation.device")
    existing = [
        path
        for name in cities
        if (path := layout.evaluation_dir / name).exists()
    ]
    if existing and not force:
        raise typer.BadParameter(
            f"refusing to overwrite existing city outputs: {existing[0]} "
            "(pass --force)"
        )
    with (
        bad_parameter_on(FileExistsError, OSError, RuntimeError, ValueError),
        cli_run(
            cfg,
            phase="evaluation",
            output_dir=layout.evaluation_dir,
            device=settings.device,
            seed=settings.seed,
            log_level=log_level,
            log_file=log_file,
        ) as run,
    ):
        run.result = evaluate_cities(settings, layout, cities, force=force)
    echo_json(run.result)


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
    city_id: Annotated[
        int | None,
        typer.Option(
            "--city-id",
            help="City to sample; optional for a single-city planner.",
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
        ("--route-temperature", route_temperature),
    ):
        if value is not None and value <= 0:
            raise typer.BadParameter(f"{name} must be positive")

    from semoflow.planner.sampling_run import sample_occurrence_planner

    with bad_parameter_on(ValueError, FileExistsError):
        result = sample_occurrence_planner(
            checkpoint_path=checkpoint,
            output_path=output,
            num_samples=num_samples,
            city_id=city_id,
            batch_size=batch_size,
            device=device,
            seed=seed,
            temperature=temperature,
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
    "compose",
    help="Compose planner samples with the latent-flow suffix realizer.",
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
            help="Occurrence planner samples from sample-plans.",
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
        Literal["first", "train_prior_argmax", "train_prior_sample"],
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
    catalog_noise_coupling: Annotated[
        Literal["independent", "trajectory_prefix"],
        typer.Option(
            "--catalog-noise-coupling",
            help=(
                "Calibrated catalog sampling noise: independent (per event) "
                "or trajectory_prefix (shared by a trajectory's events with "
                "the same planned prefix; per-event marginals unchanged)."
            ),
        ),
    ] = "independent",
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
            catalog_noise_coupling=catalog_noise_coupling,
        )

    echo_json(
        {"output": output, "rows": report.get("rows"), **report["summary"]}
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
