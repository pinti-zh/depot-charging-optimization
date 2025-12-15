import json
from math import gcd

import click
import pandas as pd
from tqdm import tqdm

from depot_charging_optimization.config import FileConfig, ModelPredictiveControlConfig, OptimizerConfig
from depot_charging_optimization.controller import policy_from_solution
from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.environment import Environment
from depot_charging_optimization.logging import get_logger, suppress_stdout_stderr
from depot_charging_optimization.optimizer.base import Optimizer
from depot_charging_optimization.optimizer.casadi import CasadiOptimizer
from depot_charging_optimization.optimizer.gurobi import GurobiOptimizer


def build_optimizer(optimizer_config: OptimizerConfig, input_data: Input) -> Optimizer | None:
    match optimizer_config.optimizer_type:
        case "casadi":
            return CasadiOptimizer(input_data, config=optimizer_config)
        case "gurobi":
            return GurobiOptimizer(input_data, config=optimizer_config)
        case _:
            return None


@click.command()
# general options
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
@FileConfig.as_click_options
@OptimizerConfig.as_click_options
@ModelPredictiveControlConfig.as_click_options
def main(
    debug: bool,
    file_config_cli_arguments: dict,
    mpc_config_cli_arguments: dict,
    optimizer_config_cli_arguments: dict,
):
    if debug:
        logger = get_logger(name="mpc", level="debug")
    else:
        logger = get_logger(name="mpc", level="info")

    if not file_config_cli_arguments["config_file"].exists():
        logger.warning(f"File config file {file_config_cli_arguments['config_file']} not found")

    if not mpc_config_cli_arguments["config_file"].exists():
        logger.warning(f"MPC config file {mpc_config_cli_arguments['config_file']} not found")

    if not optimizer_config_cli_arguments["config_file"].exists():
        logger.warning(f"Optimizer config file {optimizer_config_cli_arguments['config_file']} not found")

    file_config = FileConfig.load_from_dict(file_config_cli_arguments)
    mpc_config = ModelPredictiveControlConfig.load_from_dict(mpc_config_cli_arguments)
    optimizer_config = OptimizerConfig.load_from_dict(optimizer_config_cli_arguments)

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
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
    energy_price["energy_price"] /= 3.6e6   # convert to CHF / Joule

    grid_tariff = pd.read_csv(file_config.grid_tariff_file)
    grid_tariff["grid_tariff"] /= (365 * 1.0e6)   # convert to CHF / Watt

    dt = gcd(plan.maximum_possible_equal_timestep(), mpc_config.minutes_until_reoptimization * 60)
    plan = plan.equalize_timesteps(dt=dt)
    steps_until_reoptimization = (mpc_config.minutes_until_reoptimization * 60) // dt
    logger.debug(f"Equalized timesteps to {dt} seconds")
    logger.debug(f"Reoptimizing after {mpc_config.minutes_until_reoptimization * 60} seconds")

    plan = plan.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    plan = plan.add_grid_tariff(grid_tariff["grid_tariff"][0])

    optimizer = build_optimizer(optimizer_config, plan)
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

    def charging_efficiency(p):
        return optimizer_config.max_efficiency * (p - optimizer_config.alpha * p**2 / (2 * plan.max_charging_power))

    env = Environment(plan, initial_soe, charging_efficiency, sigma=mpc_config.mpc_energy_std_dev)
    looped_plan = plan.loop(mpc_config.num_days)

    charging_power: list[list[float]] = [[] for _ in range(plan.num_vehicles)]
    effective_charging_power: list[list[float]] = [[] for _ in range(plan.num_vehicles)]
    state_of_energy = [[initial_soe[vehicle]] for vehicle in range(plan.num_vehicles)]

    num_steps = len(plan.time) * mpc_config.num_days
    energy_cost = 0
    max_charging_power = 0
    k = 0
    policy = None
    current_soe = initial_soe

    logger.info("Running simulation")
    step_generator = range(num_steps) if debug else tqdm(range(num_steps))
    for i in step_generator:
        logger.debug(f"Step {i + 1} (t={i * dt})")
        if any(
                (soe is not None and soe < 0.0) for soe, cap in zip(current_soe, plan.battery_capacity)
        ):
            logger.warning("  [orange1]Invalid state encountered -- stopping early")
            break

        # optimize and find policy
        if k == 0:
            logger.debug(f"  [light_sea_green]Optimizing the next {steps_until_reoptimization} steps")
            optimizer_config.initial_soe = current_soe
            optimizer = build_optimizer(optimizer_config, env.plan)
            assert optimizer is not None
            optimizer.build()
            with suppress_stdout_stderr():
                solution = optimizer.solve()
            if solution is None:
                logger.warning("  [orange1]Optimizer encountered infeasible problem -- stopping early")
                break
            policy = policy_from_solution(solution, steps_until_reoptimization)
        logger.debug(
            f"  Current SoE: ({', '.join([f'{soe:.5f}' if soe is not None else '---' for soe in current_soe])[:-1]})"
        )
        logger.debug(f"  Policy: ({', '.join([f'{cp:.5f}' for cp in policy[k]])[:-1]})")

        # track energy cost and max charging power
        energy_cost += sum(cp * env.plan.time[0] * env.plan.energy_price[0] for cp in policy[k])
        max_charging_power = max(max_charging_power, max(policy[k]))

        for vehicle, cp in enumerate(policy[k]):
            charging_power[vehicle].append(cp)
            effective_charging_power[vehicle].append(charging_efficiency(cp))

        # update state of energy
        current_soe = env.step(policy[k])
        for vehicle, soe in enumerate(current_soe):
            state_of_energy[vehicle].append(env.soe[vehicle])

        k = (k + 1) % steps_until_reoptimization
        logger.debug("----------------------------------------------------------------------")

    power_cost = 1.3e-4 * max_charging_power * mpc_config.num_days
    total_cost = energy_cost + power_cost

    total_cost_str = f"{total_cost:.3f} $"
    energy_cost_str = f"{energy_cost:.3f} $"
    power_cost_str = f"{power_cost:.3f} $"
    max_cost_string_length = max(map(len, [total_cost_str, energy_cost_str, power_cost_str]))
    logger.info(f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost_str))}{total_cost_str}")
    logger.info(f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost_str))}{energy_cost_str}")
    logger.info(f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost_str))}{power_cost_str}")

    if len(charging_power[0]) < looped_plan.num_timesteps:
        looped_plan = looped_plan.truncate(len(charging_power[0]))

    solution = Solution(
        input_data=looped_plan,
        total_cost=total_cost,
        energy_cost=energy_cost,
        power_cost=power_cost,
        max_charging_power_used=max_charging_power,
        charging_power=charging_power,
        effective_charging_power=effective_charging_power,
        state_of_energy=state_of_energy,
        lower_soe_envelope=state_of_energy,
    )

    with open(file_config.solution_file, "w") as f:
        f.write(solution.model_dump_json(indent=4))
    logger.info(f"Saved solution to [cyan3]{file_config.solution_file}")
