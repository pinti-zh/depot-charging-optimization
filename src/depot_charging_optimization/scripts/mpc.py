import json
import os
from math import gcd

import click
import pandas as pd
from tqdm import tqdm

from depot_charging_optimization.config import (
    EnvironmentConfig,
    FileConfig,
    ModelPredictiveControlConfig,
    OptimizerConfig,
)
from depot_charging_optimization.controller import policy_from_solution
from depot_charging_optimization.data_models import Input
from depot_charging_optimization.environment import Environment
from depot_charging_optimization.logging import get_logger, suppress_stdout_stderr
from depot_charging_optimization.optimizer.base import Optimizer
from depot_charging_optimization.optimizer.casadi import CasadiOptimizer
from depot_charging_optimization.optimizer.gurobi import GurobiOptimizer


def build_optimizer(optimizer_config: OptimizerConfig, env_config: EnvironmentConfig, input_data: Input) -> Optimizer | None:
    match optimizer_config.optimizer_type:
        case "casadi":
            return CasadiOptimizer(input_data, config=optimizer_config, env_config=env_config)
        case "gurobi":
            return GurobiOptimizer(input_data, config=optimizer_config, env_config=env_config)
        case _:
            return None


def run_main(
    debug: bool,
    file_config: FileConfig,
    env_config: EnvironmentConfig,
    mpc_config: ModelPredictiveControlConfig,
    optimizer_config: OptimizerConfig,
):
    if debug:
        logger = get_logger(name="mpc", level="debug")
    else:
        logger = get_logger(name="mpc", level="info")

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
    logger.debug("Environment Config:")
    logger.debug(env_config)
    logger.debug("MPC Config:")
    logger.debug(mpc_config)
    logger.debug("Optimizer Config:")
    logger.debug(optimizer_config)

    input_data = []
    for data_file in file_config.data_files:
        with open(data_file, "r") as f:
            input_data.append(Input.model_validate(json.load(f)))
    plan = Input.combine(input_data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6  # convert to CHF / Joule

    grid_tariff = pd.read_csv(file_config.grid_tariff_file)
    grid_tariff["grid_tariff"] /= 365 * 1.0e6  # convert to CHF / Watt

    dt = gcd(plan.maximum_possible_equal_timestep(), mpc_config.minutes_until_reoptimization * 60)
    plan = plan.equalize_timesteps(dt=dt)
    steps_until_reoptimization = (mpc_config.minutes_until_reoptimization * 60) // dt
    logger.debug(f"Equalized timesteps to {dt} seconds")
    logger.debug(f"Reoptimizing after {mpc_config.minutes_until_reoptimization * 60} seconds")

    plan = plan.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    plan = plan.add_grid_tariff(grid_tariff["grid_tariff"][0])

    optimizer = build_optimizer(optimizer_config, env_config, plan)
    if optimizer is None:
        logger.error(f"Unknown optimizer type: {optimizer_config.optimizer_type}")
        return
    optimizer.build()

    # Get optimal initial state
    with suppress_stdout_stderr():
        global_solution = optimizer.solve()

    if global_solution is None:
        logger.error("Optimizer failed to find an initial global solution")
        return
    initial_soe = [soe[0] for soe in global_solution.state_of_energy]

    env = Environment(plan, config=env_config)
    env.reset(initial_soe)

    logger.info("Running simulation")
    step_generator = range(env.plan.num_timesteps) if debug else tqdm(range(env.plan.num_timesteps))
    k = 0
    policy = []
    for i in step_generator:
        logger.debug(f"Step {i + 1} (t={i * dt})")
        assert env.state is not None
        if not env.state.is_valid():
            logger.warning("  [orange1]Invalid state encountered -- stopping early")
            logger.warning(env.state)
            break

        # optimize and find policy
        if k == 0:
            logger.debug(f"  [light_sea_green]Optimizing the next {steps_until_reoptimization} steps")
            optimizer_config.initial_soe = env.state.state_of_energy
            optimizer = build_optimizer(optimizer_config, env_config, plan)
            assert optimizer is not None
            optimizer.build()
            with suppress_stdout_stderr():
                solution = optimizer.solve()
            if solution is None:
                logger.warning("  [orange1]Optimizer encountered infeasible problem -- stopping early")
                break
            policy = policy_from_solution(solution, steps_until_reoptimization)
            assert len(policy) == steps_until_reoptimization
        env.step(policy[k])
        plan = plan.rotate()
        logger.debug(
            f"  Current SoE: ({', '.join([f'{soe:.5f}' if soe is not None else '---' for soe in env.state.state_of_energy])[:-1]})"
        )
        logger.debug(f"  Policy: ({', '.join([f'{cp:.5f}' for cp in policy[k]])[:-1]})")

        k = (k + 1) % steps_until_reoptimization
        logger.debug("----------------------------------------------------------------------")

    solution = env.get_solution()

    # print solution
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


@click.command()
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
@FileConfig.as_click_options
@EnvironmentConfig.as_click_options
@OptimizerConfig.as_click_options
@ModelPredictiveControlConfig.as_click_options
def main(
    debug: bool,
    file_config_cli_arguments: dict,
    env_config_cli_arguments: dict,
    mpc_config_cli_arguments: dict,
    optimizer_config_cli_arguments: dict,
):
    file_config = FileConfig.load_from_dict(file_config_cli_arguments)
    env_config = EnvironmentConfig.load_from_dict(env_config_cli_arguments)
    mpc_config = ModelPredictiveControlConfig.load_from_dict(mpc_config_cli_arguments)
    optimizer_config = OptimizerConfig.load_from_dict(optimizer_config_cli_arguments)

    return run_main(debug, file_config, env_config, mpc_config, optimizer_config)
