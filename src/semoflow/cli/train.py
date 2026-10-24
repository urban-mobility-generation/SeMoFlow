from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated

import typer

from semoflow.cli.common import (
    ConfigOpt,
    DryRunConfigOpt,
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

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Train a model or tokenizer.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _torchrun_compat(
    local_rank: Annotated[
        int | None,
        typer.Option(
            "--local-rank",
            "--local_rank",
            hidden=True,
            help="torchrun-injected local process rank.",
        ),
    ] = None,
) -> None:
    if local_rank is None:
        return
    env_value = os.environ.get("LOCAL_RANK")
    if env_value is not None and int(env_value) != local_rank:
        raise typer.BadParameter(
            "--local-rank disagrees with the LOCAL_RANK environment variable"
        )
    os.environ["LOCAL_RANK"] = str(local_rank)


@app.command(
    "text-encoder", help="Encode POI text metadata into dense embeddings."
)
def text_encoder_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.encoder.pipeline import run_text_encoder

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    with cli_run(
        cfg,
        phase="text-encoder",
        output_dir=cfg.output.output_dir,
        config_phase="encoder",
        device=cfg.encoder_pipeline.text.device,
        log_level=log_level,
        log_file=log_file,
    ):
        run_text_encoder(cfg)


@app.command(
    "graph-encoder",
    help="Build graph/CF POI embeddings from mobility transitions.",
)
def graph_encoder_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    dry_run_config: DryRunConfigOpt = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.encoder.pipeline import run_graph_encoder

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    if dry_run_config:
        typer.echo(cfg.to_yaml("encoder"))
        return
    graph_cfg = cfg.encoder_pipeline.graph
    with cli_run(
        cfg,
        phase="graph-encoder",
        output_dir=Path(graph_cfg.out_dir),
        config_phase="encoder",
        device=graph_cfg.device,
        seed=graph_cfg.seed,
        log_level=log_level,
        log_file=log_file,
    ) as run:
        run.result = run_graph_encoder(cfg)


@app.command("encoder", help="Run the text encoder, then the graph encoder.")
def train_encoder_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.encoder.pipeline import run_encoder_pipeline

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    graph_cfg = cfg.encoder_pipeline.graph
    with cli_run(
        cfg,
        phase="encoder",
        output_dir=cfg.output.output_dir,
        device=(
            graph_cfg.device
            if graph_cfg.enabled
            else cfg.encoder_pipeline.text.device
        ),
        seed=graph_cfg.seed if graph_cfg.enabled else None,
        log_level=log_level,
        log_file=log_file,
    ) as run:
        run.result = run_encoder_pipeline(cfg)


@app.command(
    "tokenizer",
    help="Fit the RQ-VAE Semantic-ID tokenizer and publish its artifacts.",
)
def train_tokenizer_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.config.artifacts import TokenizerArtifacts
    from semoflow.training.tokenizer_run import run_semantic_id_tokenizer

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    tcfg = cfg.tokenizer_config
    run_dir = tcfg.run_dir()
    if TokenizerArtifacts.from_run_dir(run_dir).is_complete() and not force:
        logger.info("semantic tokenizer run already complete: %s", run_dir)
        return
    with cli_run(
        cfg,
        phase="tokenizer",
        output_dir=run_dir,
        device=tcfg.device,
        seed=tcfg.seed,
        log_level=log_level,
        log_file=log_file,
    ) as run:
        run.result = {
            "augmented_processed_visits": run_semantic_id_tokenizer(cfg)
        }


def _section[T](section: T | None, name: str) -> T:
    if section is None:
        raise typer.BadParameter(f"--config must define a {name!r} section")
    return section


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


@contextmanager
def _torchrun_process_group() -> Generator[str | None, None, None]:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        yield None
        return
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise typer.BadParameter("distributed realizer training requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    try:
        yield f"cuda:{local_rank}"
    finally:
        dist.destroy_process_group()


@app.command(
    "realizer",
    help=(
        "Train the hard-prefix latent-flow realizer from the config's "
        "realizer section. Launch with torchrun for multi-GPU DDP."
    ),
)
def train_realizer_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    dry_run_config: DryRunConfigOpt = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.realizer.training import (
        train_prefix_time_suffix_latent_flow_realizer,
    )

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    settings = _section(cfg.realizer, "realizer")
    if dry_run_config:
        typer.echo(cfg.to_yaml("realizer"))
        return
    with (
        bad_parameter_on(OSError, RuntimeError, ValueError),
        _torchrun_process_group() as rank_device,
    ):
        if rank_device is not None:
            settings = replace(settings, device=rank_device)
        with cli_run(
            cfg,
            phase="realizer",
            output_dir=settings.output_dir,
            device=settings.device,
            seed=settings.seed,
            log_level=log_level,
            log_file=log_file,
        ) as run:
            result = train_prefix_time_suffix_latent_flow_realizer(settings)
            run.result = {
                "checkpoint_path": settings.output_dir / "best.pt",
                "metrics_path": settings.output_dir / "metrics.json",
                "best_val_loss": result["best_val_loss"],
            }
    if _is_rank_zero():
        echo_json(result)


@app.command(
    "planner",
    help=(
        "Train the occurrence SID-prefix planner from the config's planner "
        "section. training.device '[0,1,...]' spawns one DDP rank per GPU."
    ),
)
def train_planner_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    dry_run_config: DryRunConfigOpt = False,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.planner.training import (
        check_planner_settings,
        train_planner,
    )

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    settings = _section(cfg.planner, "planner")
    if dry_run_config:
        typer.echo(cfg.to_yaml("planner"))
        return
    device = settings.training.device
    if not device.strip().startswith("["):
        validate_device(device, name="planner.training.device")
    with bad_parameter_on(ValueError, FileExistsError):
        # Before the run is recorded, so a refused run leaves the existing
        # outputs and their manifest untouched.
        check_planner_settings(settings, force=force)
        with cli_run(
            cfg,
            phase="planner",
            output_dir=settings.output_dir,
            device=device,
            seed=settings.seed,
            log_level=log_level,
            log_file=log_file,
        ) as run:
            result = train_planner(settings, force=force, log_level=log_level)
            run.result = asdict(result)
    if _is_rank_zero():
        echo_json(run.result)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
