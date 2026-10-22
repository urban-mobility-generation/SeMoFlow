from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
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
from semoflow.logging.primitives import configure_logging

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
        config_phase="encoder",
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
        typer.echo(cfg.to_yaml())
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
    ) as state:
        state.result = run_graph_encoder(cfg)


@app.command("encoder", help="Run the text encoder, then the graph encoder.")
def train_encoder_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.encoder.pipeline import (
        run_encoder_pipeline,
    )

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    with cli_run(
        cfg, phase="encoder", log_level=log_level, log_file=log_file
    ) as state:
        state.result = run_encoder_pipeline(cfg)


@app.command(
    "tokenizer",
    help=("Fit the RQ-VAE Semantic-ID tokenizer and publish its artifacts."),
)
def train_tokenizer_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild even if output already exists."),
    ] = False,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.training.tokenizer_run import (
        run_semantic_id_tokenizer,
    )

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    run_semantic_id_tokenizer(
        cfg, force=force, log_level=log_level, log_file=log_file
    )


def _parse_planner_devices(device: str) -> tuple[str, list[int]]:
    import torch

    from semoflow.training.utils.runtime import (
        parse_device_spec,
    )

    try:
        primary, device_ids = parse_device_spec(device)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(f"invalid --device: {device!r}") from exc
    if not device_ids:
        validate_device(device)
        return str(primary), []
    if not torch.cuda.is_available():
        raise typer.BadParameter(
            "--device requests CUDA GPUs, but CUDA is unavailable"
        )
    device_count = torch.cuda.device_count()
    if len(set(device_ids)) != len(device_ids) or any(
        device_id < 0 or device_id >= device_count for device_id in device_ids
    ):
        raise typer.BadParameter(
            f"--device GPU list {device_ids} is outside the "
            f"{device_count} visible CUDA devices or contains duplicates"
        )
    return device, device_ids


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


@app.command("realizer", help="Train the hard-prefix latent-flow realizer.")
def train_latent_flow_realizer_cmd(
    bank: Annotated[
        Path,
        typer.Option(
            "--bank",
            help="Bank from `semoflow-artifacts build-realizer-bank`.",
        ),
    ],
    codebooks: Annotated[
        Path, typer.Option("--codebooks", help="Tokenizer codebook bundle.")
    ],
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Run output directory.")
    ],
    initialize_from: Annotated[
        Path | None, typer.Option("--initialize-from")
    ] = None,
    prefix_depth: Annotated[int, typer.Option("--prefix-depth")] = 3,
    codebook_size: Annotated[int, typer.Option("--codebook-size")] = 64,
    time_vocab: Annotated[int, typer.Option("--time-vocab")] = 42,
    latent_dim: Annotated[int | None, typer.Option("--latent-dim")] = None,
    model_dim: Annotated[int, typer.Option("--model-dim")] = 256,
    condition_depth: Annotated[int, typer.Option("--condition-depth")] = 4,
    depth: Annotated[int, typer.Option("--depth")] = 6,
    num_heads: Annotated[int, typer.Option("--num-heads")] = 8,
    mlp_ratio: Annotated[float, typer.Option("--mlp-ratio")] = 4.0,
    dropout: Annotated[float, typer.Option("--dropout")] = 0.1,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 256,
    epochs: Annotated[int, typer.Option("--epochs")] = 30,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 2e-4,
    lr_scheduler: Annotated[str, typer.Option("--lr-scheduler")] = "constant",
    warmup_steps: Annotated[int, typer.Option("--warmup-steps")] = 0,
    min_lr: Annotated[float, typer.Option("--min-lr")] = 0.0,
    early_stopping_patience: Annotated[
        int | None, typer.Option("--early-stopping-patience")
    ] = None,
    early_stopping_min_delta: Annotated[
        float, typer.Option("--early-stopping-min-delta")
    ] = 0.0,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 1e-2,
    max_grad_norm: Annotated[float, typer.Option("--max-grad-norm")] = 1.0,
    train_max_samples: Annotated[
        int | None, typer.Option("--train-max-samples")
    ] = None,
    val_max_samples: Annotated[
        int | None, typer.Option("--val-max-samples")
    ] = None,
    num_workers: Annotated[int, typer.Option("--num-workers")] = 0,
    prefetch_factor: Annotated[int, typer.Option("--prefetch-factor")] = 2,
    cpu_threads: Annotated[int, typer.Option("--cpu-threads")] = 4,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    compile_model: Annotated[
        bool, typer.Option("--compile/--no-compile")
    ] = False,
    mixed_precision: Annotated[
        str, typer.Option("--mixed-precision")
    ] = "none",
    tf32: Annotated[bool, typer.Option("--tf32/--no-tf32")] = False,
    fused_optimizer: Annotated[
        bool, typer.Option("--fused-optimizer/--no-fused-optimizer")
    ] = False,
    val_every: Annotated[int, typer.Option("--val-every")] = 1,
    uniform_t: Annotated[
        bool, typer.Option("--uniform-t/--lognormal-t")
    ] = False,
    city_vocab: Annotated[int, typer.Option("--city-vocab")] = 0,
    spatial_conditioning: Annotated[
        bool, typer.Option("--spatial-conditioning/--no-spatial-conditioning")
    ] = False,
    spatial_num_basis: Annotated[int, typer.Option("--spatial-num-basis")] = 8,
    spatial_max_km: Annotated[float, typer.Option("--spatial-max-km")] = 50.0,
    places_path: Annotated[Path | None, typer.Option("--places-path")] = None,
    prefix_embedding_mode: Annotated[
        str, typer.Option("--prefix-embedding-mode")
    ] = "learned",
    tail_representation: Annotated[
        str, typer.Option("--tail-representation")
    ] = "summed",
    latent_target_scope: Annotated[
        str, typer.Option("--latent-target-scope")
    ] = "suffix",
    proposal_corruption_rate: Annotated[
        float, typer.Option("--proposal-corruption-rate")
    ] = 0.0,
    tail_normalization: Annotated[
        str, typer.Option("--tail-normalization")
    ] = "diagonal",
    tail_whitening_shrinkage: Annotated[
        float, typer.Option("--tail-whitening-shrinkage")
    ] = 0.05,
    tail_whitening_chunk_size: Annotated[
        int, typer.Option("--tail-whitening-chunk-size")
    ] = 262144,
    catalog_decoder_mode: Annotated[
        str, typer.Option("--catalog-decoder-mode")
    ] = "nearest",
    factorized_sum_consistency_weight: Annotated[
        float, typer.Option("--factorized-sum-consistency-weight")
    ] = 0.0,
    catalog_path: Annotated[
        Path | None, typer.Option("--catalog-path")
    ] = None,
    catalog_loss_weight: Annotated[
        float, typer.Option("--catalog-loss-weight")
    ] = 0.0,
    catalog_negative_count: Annotated[
        int, typer.Option("--catalog-negative-count")
    ] = 8,
    catalog_temperature: Annotated[
        float, typer.Option("--catalog-temperature")
    ] = 1.0,
    catalog_candidate_count: Annotated[
        int, typer.Option("--catalog-candidate-count")
    ] = 64,
    endpoint_loss_weight: Annotated[
        float, typer.Option("--endpoint-loss-weight")
    ] = 0.0,
    population_loss_weight: Annotated[
        float, typer.Option("--population-loss-weight")
    ] = 0.0,
    catalog_prior_weight: Annotated[
        float, typer.Option("--catalog-prior-weight")
    ] = 0.25,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    from semoflow.realizer.training import (
        PrefixTimeSuffixLatentFlowRealizerConfig,
        train_prefix_time_suffix_latent_flow_realizer,
    )

    with (
        bad_parameter_on(OSError, RuntimeError, ValueError),
        _torchrun_process_group() as rank_device,
    ):
        config = PrefixTimeSuffixLatentFlowRealizerConfig(
            bank_path=bank,
            codebooks_path=codebooks,
            output_dir=output_dir,
            initialize_from=initialize_from,
            prefix_depth=prefix_depth,
            codebook_size=codebook_size,
            time_vocab=time_vocab,
            latent_dim=latent_dim,
            model_dim=model_dim,
            condition_depth=condition_depth,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            train_max_samples=train_max_samples,
            val_max_samples=val_max_samples,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            cpu_threads=cpu_threads,
            device=rank_device or device,
            seed=seed,
            compile=compile_model,
            mixed_precision=mixed_precision,
            tf32=tf32,
            fused_optimizer=fused_optimizer,
            val_every=val_every,
            lognormal_t=not uniform_t,
            city_vocab=city_vocab,
            spatial_conditioning=spatial_conditioning,
            spatial_num_basis=spatial_num_basis,
            spatial_max_km=spatial_max_km,
            places_path=places_path,
            prefix_embedding_mode=prefix_embedding_mode,
            tail_representation=tail_representation,
            latent_target_scope=latent_target_scope,
            proposal_corruption_rate=proposal_corruption_rate,
            tail_normalization=tail_normalization,
            tail_whitening_shrinkage=tail_whitening_shrinkage,
            tail_whitening_chunk_size=tail_whitening_chunk_size,
            catalog_decoder_mode=catalog_decoder_mode,
            factorized_sum_consistency_weight=factorized_sum_consistency_weight,
            catalog_path=catalog_path,
            catalog_loss_weight=catalog_loss_weight,
            catalog_negative_count=catalog_negative_count,
            catalog_temperature=catalog_temperature,
            catalog_candidate_count=catalog_candidate_count,
            endpoint_loss_weight=endpoint_loss_weight,
            population_loss_weight=population_loss_weight,
            catalog_prior_weight=catalog_prior_weight,
        )
        result = train_prefix_time_suffix_latent_flow_realizer(config)
    if _is_rank_zero():
        echo_json(result)


@app.command(
    "planner",
    help=("Train the occurrence SID-prefix planner from a dataset artifact."),
)
def train_planner_cmd(
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset", help="Occurrence planner dataset artifact (.pt)."
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory for best.pt, last.pt, and metrics.json.",
        ),
    ],
    model_dim: Annotated[int, typer.Option("--model-dim")] = 128,
    n_heads: Annotated[int, typer.Option("--n-heads")] = 4,
    n_layers: Annotated[int, typer.Option("--n-layers")] = 4,
    dropout: Annotated[float, typer.Option("--dropout")] = 0.1,
    prefix_decoder: Annotated[
        str, typer.Option("--prefix-decoder", help="'hierarchical' or 'flat'.")
    ] = "hierarchical",
    tokenizer_latent: Annotated[
        bool,
        typer.Option(
            "--tokenizer-latent/--no-tokenizer-latent",
            help=(
                "Add the frozen tokenizer codebook reconstruction to the SID "
                "history embedding. Requires --codebooks at dataset build."
            ),
        ),
    ] = False,
    distance_prior: Annotated[
        bool,
        typer.Option(
            "--distance-prior/--no-distance-prior",
            help=(
                "Bias the Explore prefix decoder by great-circle distance "
                "from the previous event. Requires --places coordinates."
            ),
        ),
    ] = False,
    mobility_router: Annotated[
        str,
        typer.Option(
            "--mobility-router",
            help="'none' or 'explore_retrieve'.",
        ),
    ] = "none",
    mobility_gate_prior: Annotated[
        str,
        typer.Option(
            "--mobility-gate-prior",
            help=(
                "Explore/Retrieve gate prior over the distinct-prefix count: "
                "'log_distinct', 'epr', or 'none'."
            ),
        ),
    ] = "log_distinct",
    epochs: Annotated[int, typer.Option("--epochs")] = 10,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 64,
    learning_rate: Annotated[float, typer.Option("--learning-rate")] = 0.002,
    weight_decay: Annotated[float, typer.Option("--weight-decay")] = 0.0001,
    lr_scheduler: Annotated[
        str, typer.Option("--lr-scheduler", help="'constant' or 'cosine'.")
    ] = "constant",
    warmup_steps: Annotated[
        int, typer.Option("--warmup-steps", help="Linear warmup update steps.")
    ] = 0,
    min_lr: Annotated[
        float,
        typer.Option(
            "--min-lr", help="Minimum learning rate for cosine scheduler."
        ),
    ] = 0.0,
    length_weight: Annotated[float, typer.Option("--length-weight")] = 1.0,
    gap_weight: Annotated[
        float,
        typer.Option(
            "--gap-weight",
            help="Weight for gap-bucket NLL plus continuous-duration NLL.",
        ),
    ] = 1.0,
    prefix_weight: Annotated[float, typer.Option("--prefix-weight")] = 1.0,
    recurrence_weight: Annotated[
        float,
        typer.Option(
            "--recurrence-weight",
            help=(
                "Auxiliary causal router loss for repeated SID prefixes; 0 disables it."
            ),
        ),
    ] = 0.0,
    label_smoothing: Annotated[
        float,
        typer.Option(
            "--label-smoothing",
            help=(
                "Prefix label smoothing. Must stay 0 when "
                "--mobility-router is not 'none'."
            ),
        ),
    ] = 0.0,
    num_workers: Annotated[
        int,
        typer.Option(
            "--num-workers",
            help="Number of background data loading workers (default: 0).",
        ),
    ] = 0,
    early_stopping_patience: Annotated[
        int,
        typer.Option(
            "--early-stopping-patience",
            help=(
                "Stop after this many epochs without a validation-NLL "
                "improvement (default: 0, disabled)."
            ),
        ),
    ] = 0,
    distributed: Annotated[
        bool,
        typer.Option(
            "--distributed/--no-distributed",
            help="Enable torchrun DistributedDataParallel training.",
        ),
    ] = False,
    compile_model: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile",
            help="Compile the planner training model with torch.compile.",
        ),
    ] = False,
    compile_backend: Annotated[
        str,
        typer.Option(
            "--compile-backend",
            help="torch.compile backend (default: inductor).",
        ),
    ] = "inductor",
    compile_mode: Annotated[
        str,
        typer.Option(
            "--compile-mode",
            help=(
                "torch.compile mode: default, reduce-overhead, max-autotune, "
                "or max-autotune-no-cudagraphs."
            ),
        ),
    ] = "default",
    device: Annotated[
        str,
        typer.Option(
            "--device",
            help=(
                "Compute device: cpu/cuda:N for one process, '[0,1,2,3]' "
                "to spawn DDP, or cuda with torchrun --distributed."
            ),
        ),
    ] = "cpu",
    seed: Annotated[int, typer.Option("--seed")] = 0,
    initialize_from: Annotated[
        Path | None,
        typer.Option(
            "--initialize-from",
            help=(
                "Warm-start neural weights from a source planner checkpoint "
                "with identical architecture. The catalog and city set may "
                "differ; city rows are matched by city ID and a city the "
                "source never saw starts from a zero row. The optimizer, "
                "schedule, and epoch counters still start fresh."
            ),
        ),
    ] = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    configure_logging(level=log_level)
    normalized_device, device_ids = _parse_planner_devices(device)
    from semoflow.planner.artifacts import (
        OccurrencePlannerTrainingOptions,
    )
    from semoflow.planner.training import (
        OccurrencePlannerArchitecture,
        train_occurrence_planner,
        train_occurrence_planner_on_devices,
    )
    from semoflow.planner.warm_start import PlannerWarmStart

    with bad_parameter_on(ValueError, FileExistsError):
        warm_start = (
            None
            if initialize_from is None
            else PlannerWarmStart(checkpoint=initialize_from)
        )
        architecture = OccurrencePlannerArchitecture(
            model_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            prefix_decoder=prefix_decoder,
            use_tokenizer_latent=tokenizer_latent,
            use_distance_prior=distance_prior,
            mobility_router=mobility_router,
            mobility_gate_prior=mobility_gate_prior,
        )
        options = OccurrencePlannerTrainingOptions(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            length_weight=length_weight,
            gap_weight=gap_weight,
            prefix_weight=prefix_weight,
            recurrence_weight=recurrence_weight,
            label_smoothing=label_smoothing,
            device=normalized_device,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            min_lr=min_lr,
            num_workers=num_workers,
            early_stopping_patience=early_stopping_patience,
            distributed=distributed,
            compile_model=compile_model,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
        )
        if len(device_ids) > 1 and not distributed:
            result = train_occurrence_planner_on_devices(
                dataset_path=dataset,
                output_dir=output_dir,
                architecture=architecture,
                options=options,
                seed=seed,
                device_ids=device_ids,
                log_level=log_level,
                force=force,
                warm_start=warm_start,
            )
        else:
            result = train_occurrence_planner(
                dataset_path=dataset,
                output_dir=output_dir,
                architecture=architecture,
                options=options,
                seed=seed,
                force=force,
                warm_start=warm_start,
            )
    if _is_rank_zero():
        echo_json(
            {
                "output_dir": result.output_dir,
                "best_checkpoint": result.best_checkpoint,
                "last_checkpoint": result.last_checkpoint,
                "metrics_path": result.metrics_path,
                "best_epoch": result.best_epoch,
                "best_validation_nll": result.best_validation_nll,
                "final_train_objective": result.final_train_objective,
                "final_train_nll": result.final_train_nll,
                "final_validation_nll": result.final_validation_nll,
            }
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
