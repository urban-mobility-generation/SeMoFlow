from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from semoflow.cli.common import (
    ConfigOpt,
    LogFileOpt,
    LogLevelOpt,
    SetArgsOpt,
    cli_run,
    resolve_cli_config,
)

app = typer.Typer(
    help="Preprocess Foursquare check-ins into SeMoFlow artifacts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    help=(
        "Run the preprocessing pipeline on the configured dataset.\n\n"
        "Every run reads raw data and overwrites "
        "processed_visits.parquet, quality_scores.parquet, and "
        "places_index.parquet."
    )
)
def preprocess_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out-dir",
            help="Output directory. Shortcut for --set output.output_dir=<path>.",
        ),
    ] = None,
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.config.artifacts import PreprocessingArtifacts
    from semoflow.preprocess.foursquare import run_foursquare_pipeline

    overrides = list(set_args or [])
    if out_dir is not None:
        overrides.append(f"output.output_dir={out_dir}")
    cfg = resolve_cli_config(
        config, overrides, log_level=log_level, log_file=log_file
    )
    with cli_run(
        cfg,
        phase="preprocessing",
        output_dir=PreprocessingArtifacts.from_config(cfg).run_dir,
        log_level=log_level,
        log_file=log_file,
    ):
        run_foursquare_pipeline(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
