from __future__ import annotations

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
    log_level: LogLevelOpt = "INFO",
    log_file: LogFileOpt = None,
) -> None:
    from semoflow.preprocess.foursquare import run_foursquare_pipeline

    cfg = resolve_cli_config(
        config, set_args, log_level=log_level, log_file=log_file
    )
    with cli_run(
        cfg,
        phase="preprocessing",
        output_dir=cfg.layout.preprocessing.run_dir,
        log_level=log_level,
        log_file=log_file,
    ):
        run_foursquare_pipeline(cfg)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
