from __future__ import annotations

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

app = typer.Typer(
    help="Derive and assemble identity-keyed SeMoFlow artifacts.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _group() -> None:
    """Keep `build` a named subcommand even though it is the only one."""


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
