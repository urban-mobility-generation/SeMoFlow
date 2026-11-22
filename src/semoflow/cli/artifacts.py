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
    help="Build a stage's training data from the tokenizer outputs.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command(
    "planner",
    help="Build the weekly segments and the planner dataset.",
)
def planner_data_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    force: ForceOpt = False,
    log_level: LogLevelOpt = "INFO",
) -> None:
    from semoflow.planner.dataset_build import build_planner_data

    cfg = resolve_cli_config(config, set_args, log_level=log_level)
    with bad_parameter_on(FileExistsError, OSError, TypeError, ValueError):
        echo_json(
            build_planner_data(cfg.planner.data, cfg.layout, force=force)
        )


@app.command(
    "realizer",
    help="Build the realizer bank from the weekly segments.",
)
def realizer_data_cmd(
    config: ConfigOpt = None,
    set_args: SetArgsOpt = None,
    log_level: LogLevelOpt = "INFO",
) -> None:
    from semoflow.realizer.bank import build_realizer_data

    cfg = resolve_cli_config(config, set_args, log_level=log_level)
    with bad_parameter_on(FileExistsError, OSError, TypeError, ValueError):
        echo_json(build_realizer_data(cfg.realizer, cfg.layout))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
