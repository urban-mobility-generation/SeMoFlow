from __future__ import annotations

import logging
import os
import platform
import shlex
import subprocess
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from semoflow.config import ExperimentConfig, load_config
from semoflow.utils.log import configure_logging
from semoflow.utils.serialization import dump_json, write_json

logger = logging.getLogger(__name__)

app = typer.Typer(
    help=(
        "Run a SeMoFlow stage. Every stage reads one experiment config and "
        "keeps its outputs under run_dir."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

Config = Annotated[
    Path,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        help="Experiment YAML, for example configs/foursquare.yaml.",
    ),
]
Sets = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help=(
            "Override a config field, for example run_dir=/data/run or "
            "planner.training.epochs=20. May be repeated."
        ),
    ),
]
LogLevel = Annotated[
    str, typer.Option("--log-level", help="DEBUG, INFO, WARNING or ERROR.")
]
Force = Annotated[
    bool, typer.Option("--force", help="Overwrite existing outputs.")
]


def _load(
    config: Path, sets: list[str] | None, log_level: str
) -> ExperimentConfig:
    configure_logging(level=log_level)
    try:
        return load_config(config, list(sets or []))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _echo(payload: object) -> None:
    if _is_rank_zero():
        typer.echo(dump_json(payload))


_RECORDED_PACKAGES = ("torch", "polars", "numpy", "pyyaml")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_info() -> dict[str, str | bool]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": _git("status", "--porcelain") != "",
        "origin": _git("remote", "get-url", "origin"),
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in _RECORDED_PACKAGES
        },
    }


def _write_manifest(directory: Path, **fields: Any) -> None:
    argv = [str(part) for part in sys.argv]
    manifest = {
        "command": shlex.join(argv),
        "argv": argv,
        "cwd": str(Path.cwd()),
        "git": _git_info(),
        "environment": _environment_info(),
        **fields,
    }
    write_json(directory / "run_manifest.json", manifest)


@contextmanager
def _run(
    cfg: ExperimentConfig, stage: str, output_dir: Path, log_level: str
) -> Generator[None, None, None]:
    if not _is_rank_zero():
        yield
        return
    configure_logging(
        level=log_level, log_file=output_dir / "logs" / "run.log"
    )
    config_path = output_dir / "config_resolved.yaml"
    cfg.save_yaml(config_path)
    logger.info("%s started  output_dir=%s", stage, output_dir)
    started = time.perf_counter()
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        elapsed = round(time.perf_counter() - started, 1)
        _write_manifest(
            output_dir,
            experiment=cfg.name,
            stage=stage,
            config_path=config_path,
            status=status,
            elapsed_seconds=elapsed,
        )
        logger.info("%s %s in %.1fs", stage, status, elapsed)


@app.command(help="Print the resolved experiment config.")
def config(config: Config, sets: Sets = None) -> None:
    typer.echo(_load(config, sets, "WARNING").to_yaml())


@app.command(help="Turn the raw Foursquare release into visits and places.")
def preprocess(
    config: Config, sets: Sets = None, log_level: LogLevel = "INFO"
) -> None:
    from semoflow.preprocess.foursquare import run_foursquare_pipeline

    cfg = _load(config, sets, log_level)
    with _run(cfg, "preprocess", cfg.layout.preprocessing.run_dir, log_level):
        run_foursquare_pipeline(cfg)


@app.command(help="Embed POI text, then train the POI graph encoder.")
def encoder(
    config: Config, sets: Sets = None, log_level: LogLevel = "INFO"
) -> None:
    from semoflow.encoder.pipeline import run_encoder_pipeline

    cfg = _load(config, sets, log_level)
    with _run(cfg, "encoder", cfg.layout.encoder_dir, log_level):
        run_encoder_pipeline(cfg)


@app.command(
    help="Train the RQ-VAE tokenizer and assign every POI a semantic ID."
)
def tokenizer(
    config: Config,
    sets: Sets = None,
    force: Force = False,
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.training.tokenizer_run import run_semantic_id_tokenizer

    cfg = _load(config, sets, log_level)
    artifacts = cfg.layout.tokenizer
    if artifacts.is_complete() and not force:
        logger.info("tokenizer already complete: %s", artifacts.run_dir)
        return
    with _run(cfg, "tokenizer", artifacts.run_dir, log_level):
        run_semantic_id_tokenizer(cfg)


@app.command(
    "planner-data", help="Build the weekly segments and planner dataset."
)
def planner_data(
    config: Config,
    sets: Sets = None,
    force: Force = False,
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.planner.dataset_build import build_planner_data

    cfg = _load(config, sets, log_level)
    _echo(build_planner_data(cfg.planner.data, cfg.layout, force=force))


@app.command(
    help=(
        "Train the SID-prefix planner. planner.training.device='[0,1]' "
        "trains on several GPUs."
    )
)
def planner(
    config: Config,
    sets: Sets = None,
    force: Force = False,
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.planner.training import check_planner_settings, train_planner

    cfg = _load(config, sets, log_level)
    try:
        check_planner_settings(cfg.planner, cfg.layout, force=force)
    except (FileExistsError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    with _run(cfg, "planner", cfg.layout.planner_dir, log_level):
        result = train_planner(
            cfg.planner, cfg.layout, force=force, log_level=log_level
        )
    _echo(result)


@app.command(
    "realizer-data", help="Build the realizer bank from the segments."
)
def realizer_data(
    config: Config, sets: Sets = None, log_level: LogLevel = "INFO"
) -> None:
    from semoflow.realizer.bank import build_realizer_data

    cfg = _load(config, sets, log_level)
    _echo(build_realizer_data(cfg.realizer, cfg.layout))


@contextmanager
def _torchrun_device() -> Generator[str | None, None, None]:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        yield None
        return
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        yield f"cuda:{local_rank}"
    finally:
        dist.destroy_process_group()


@app.command(
    help=(
        "Train the suffix realizer. Launch under torchrun to train on "
        "several GPUs."
    )
)
def realizer(
    config: Config, sets: Sets = None, log_level: LogLevel = "INFO"
) -> None:
    from semoflow.realizer.training import (
        train_prefix_time_suffix_latent_flow_realizer,
    )

    cfg = _load(config, sets, log_level)
    settings = cfg.realizer
    with _torchrun_device() as device:
        if device is not None:
            settings = replace(settings, device=device)
        with _run(cfg, "realizer", cfg.layout.realizer_dir, log_level):
            result = train_prefix_time_suffix_latent_flow_realizer(
                settings, cfg.layout
            )
    _echo(result)


@app.command(
    help=(
        "Sample plans for each evaluation city, compose them into POI "
        "trajectories, and score them against held-out weeks."
    )
)
def evaluate(
    config: Config,
    sets: Sets = None,
    force: Force = False,
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.metrics.city_evaluation import evaluate_cities

    cfg = _load(config, sets, log_level)
    layout, cities = cfg.layout, cfg.evaluation_cities
    existing = [
        path
        for city in cities
        if (path := layout.evaluation_dir / city).exists()
    ]
    if existing and not force:
        raise typer.BadParameter(
            f"{existing[0]} exists; pass --force to overwrite"
        )
    with _run(cfg, "evaluate", layout.evaluation_dir, log_level):
        result = evaluate_cities(cfg.evaluation, layout, cities, force=force)
    _echo(result)


@app.command("sample-plans", help="Sample plans from a planner checkpoint.")
def sample_plans(
    checkpoint: Annotated[Path, typer.Option(help="Planner checkpoint.")],
    output: Annotated[Path, typer.Option(help="Output samples (.pt).")],
    city_id: Annotated[
        int | None,
        typer.Option(help="City to sample; optional for one-city planners."),
    ] = None,
    num_samples: int = 1,
    batch_size: Annotated[
        int, typer.Option(help="Plans sampled per batch.")
    ] = 1024,
    temperature: float = 1.0,
    route_temperature: Annotated[
        float | None,
        typer.Option(
            help="Explore/Retrieve gate temperature [default: --temperature]."
        ),
    ] = None,
    device: str = "cpu",
    seed: int = 0,
    force: Force = False,
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.planner.sampling_run import sample_occurrence_planner

    configure_logging(level=log_level)
    if temperature <= 0 or (
        route_temperature is not None and route_temperature <= 0
    ):
        raise typer.BadParameter("temperatures must be positive")
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
    _echo(
        {
            "output_path": result.output_path,
            "num_samples": result.num_samples,
            "batch_size": result.batch_size,
        }
    )


@app.command(help="Compose planner samples with the realizer and score them.")
def compose(
    realizer_checkpoint: Annotated[
        Path, typer.Option(help="Realizer checkpoint.")
    ],
    planner_samples: Annotated[
        Path, typer.Option(help="Output of sample-plans.")
    ],
    bank: Annotated[
        Path, typer.Option(help="Realizer bank with the reference split.")
    ],
    codebooks: Annotated[Path, typer.Option(help="Tokenizer codebooks.")],
    lookup: Annotated[Path, typer.Option(help="Semantic-ID POI lookup.")],
    grounding_index: Annotated[
        Path, typer.Option(help="Semantic-ID grounding index.")
    ],
    places: Annotated[Path, typer.Option(help="Places index.")],
    output: Annotated[Path, typer.Option(help="Output report (.json).")],
    max_samples: Annotated[
        int, typer.Option(help="Trajectories per arm.")
    ] = 10000,
    batch_size: Annotated[
        int, typer.Option(help="Realizer sampling batch.")
    ] = 64,
    device: str = "cuda:0",
    num_steps: Annotated[int, typer.Option(help="ODE solver steps.")] = 32,
    solver: Literal["euler", "heun"] = "heun",
    reference_split: Literal["val", "test"] = "val",
    candidate_selection: Annotated[
        Literal["first", "train_prior_argmax", "train_prior_sample"],
        typer.Option(help="How a generated SID picks its POI."),
    ] = "first",
    candidate_prior: Annotated[
        Path | None, typer.Option(help="Train-split POI prior.")
    ] = None,
    catalog_temperature: Annotated[
        float | None, typer.Option(help="Override the catalog temperature.")
    ] = None,
    catalog_noise_coupling: Annotated[
        Literal["independent", "trajectory_prefix"],
        typer.Option(
            help=(
                "independent draws each visit's POI separately; "
                "trajectory_prefix shares the noise among a trajectory's "
                "visits with the same planned prefix."
            )
        ),
    ] = "independent",
    log_level: LogLevel = "INFO",
) -> None:
    from semoflow.metrics.planner_realizer_composition import (
        evaluate_planner_realizer,
    )

    configure_logging(level=log_level)
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
    _echo({"output": output, "rows": report.get("rows"), **report["summary"]})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
