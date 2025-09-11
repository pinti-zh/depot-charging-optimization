import json
import os
from pathlib import Path
from time import perf_counter

import click
import pandas as pd

from depot_charging_optimization.config import FileConfig, OptimizerConfig
from depot_charging_optimization.core import CasadiOptimizer, GurobiOptimizer, Optimizer
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.logging import get_logger, log_stdout
from depot_charging_optimization.result_store import ResultStore


@click.command()
# general options
@click.argument("data-files", type=Path, nargs=-1)
@click.option("--energy-price-file", type=Path, help="energy price file")
@click.option("--solution-file", type=Path, help="solution file")
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
# optimizer options
@click.option("--optimizer-type", type=str)
@click.option("--ce-function-type", type=str)
@click.option("--alpha", type=float)
@click.option("--bidirectional-charging", is_flag=True, default=None)
@click.option("--confidence-level", type=float)
@click.option("--energy-std-dev", type=float)
def main(
    data_files: list[Path] | None,
    energy_price_file: Path | None,
    solution_file: Path | None,
    debug: bool,
    optimizer_type: str | None,
    ce_function_type: str | None,
    alpha: float | None,
    bidirectional_charging: bool | None,
    confidence_level: float | None,
    energy_std_dev: float | None,
):
    assert data_files is not None
    if len(data_files) == 0:
        data_files = None
    file_kwargs = {k: v for k, v in locals().items() if v is not None and k in FileConfig.model_fields}
    file_config = FileConfig(**file_kwargs)

    optimizer_kwargs = {k: v for k, v in locals().items() if v is not None and k in OptimizerConfig.model_fields}
    optimizer_config = OptimizerConfig(**optimizer_kwargs)
    optimize(debug, file_config, optimizer_config)


def optimize(
    debug: bool,
    file_config: FileConfig,
    optimizer_config: OptimizerConfig,
):
    if debug:
        logger = get_logger(name="optimize", level="debug")
    else:
        logger = get_logger(name="optimize", level="info")
    data = []

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
    logger.debug("Optimizer Config:")
    logger.debug(optimizer_config)

    logger.info("Reading files:")
    for i, file in enumerate(file_config.data_files):
        with open(file) as f:
            data.append(Input.model_validate(json.load(f)))
        logger.info(f"  {i + 1}. [cyan]{file}")
    logger.info("")

    data_input = Input.combine(data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    data_input = data_input.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    data_input = data_input.add_grid_tariff(1.2e-4)
    # data_input = data_input.add_grid_tariff((17.0 / 30) * 1e-3)

    # optimization
    start = perf_counter()
    optimizer: Optimizer | None = None
    if optimizer_config.optimizer_type == "casadi":
        optimizer = CasadiOptimizer(
            data_input,
            config=optimizer_config,
        )
    else:
        optimizer = GurobiOptimizer(
            data_input,
            config=optimizer_config,
        )

    optimizer.build(ce_function_type=optimizer_config.ce_function_type, alpha=optimizer_config.alpha)

    # solve
    with log_stdout(logger, level="debug"):
        solution = optimizer.solve()
    optimization_time = perf_counter() - start

    if solution is None:
        logger.error("No solution found")
    else:
        # slack information
        max_slack = 0
        max_slack_location = None
        for sublist in optimizer.slack["state_of_energy"]:
            if max(sublist) > max_slack:
                max_slack = max(sublist)
                max_slack_location = "State of Energy"
        for sublist in optimizer.slack["lower_soe_envelope"]:
            if max(sublist) > max_slack:
                max_slack = max(sublist)
                max_slack_location = "Lower SoE Envelope"
        for sublist in optimizer.slack["charging_power"]:
            if max(sublist) > max_slack:
                max_slack = max(sublist)
                max_slack_location = "Charging Power"
        max_slack = max(max_slack, optimizer.slack["max_charging_power"])
        if optimizer.slack["max_charging_power"] > max_slack:
            max_slack = optimizer.slack["max_charging_power"]
            max_slack_location = "Max Charging Power"
        logger.debug(f"Maximum slack: {max_slack:.3e} (found in [{max_slack_location}] constraints)")

        # solution information
        logger.info(f"Found solution in {optimization_time:.4f} seconds")
        total_cost = f"{solution.total_cost:.3f} $"
        energy_cost = f"{solution.energy_cost:.3f} $"
        power_cost = f"{solution.power_cost:.3f} $"
        max_cost_string_length = max(map(len, [total_cost, energy_cost, power_cost]))
        logger.info(f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost))}{total_cost}")
        logger.info(f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost))}{energy_cost}")
        logger.info(f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost))}{power_cost}")

        solution_dir = os.path.dirname(file_config.solution_file)
        os.makedirs(solution_dir, exist_ok=True)
        with open(file_config.solution_file, "w") as f:
            f.write(solution.model_dump_json(indent=4))
        logger.info(f"Saved solution to [cyan3]{file_config.solution_file}")

        result_store = ResultStore(Path("logs/optimize.log"))
        result_store.write(
            {
                "solution_total_cost": solution.total_cost,
                "input": {
                    "data_files": list(map(str, file_config.data_files)),
                    "energy_price_file": str(file_config.energy_price_file),
                    "ce_function": optimizer_config.ce_function_type,
                    "alpha": optimizer_config.alpha,
                    "solution_file": str(file_config.solution_file),
                    "bidirectional_charging": optimizer_config.bidirectional_charging,
                    "optimizer_type": optimizer_config.optimizer_type,
                    "debug": debug,
                },
            }
        )
