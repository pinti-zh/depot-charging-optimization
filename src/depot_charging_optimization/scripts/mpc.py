import json

import click
import pandas as pd
from tqdm import tqdm

from depot_charging_optimization.config import FileConfig, OptimizerConfig
from depot_charging_optimization.controller import policy_from_solution
from depot_charging_optimization.core import CasadiOptimizer, GurobiOptimizer
from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.environment import Environment
from depot_charging_optimization.logging import get_logger, suppress_stdout_stderr


def run_main(
    debug: bool,
    steps_until_reoptimization: int,
    equalize_timesteps: bool,
    days: int,
    file_config: FileConfig,
    optimizer_config: OptimizerConfig,
):
    if debug:
        logger = get_logger(name="mpc", level="debug")
    else:
        logger = get_logger(name="mpc", level="info")

    # log config
    logger.debug("File Config:")
    logger.debug(file_config)
    logger.debug("Optimizer Config:")
    logger.debug(optimizer_config)

    input_data = []
    for data_file in file_config.data_files:
        with open(data_file, "r") as f:
            input_data.append(Input.model_validate(json.load(f)))
    plan = Input.combine(input_data)

    energy_price = pd.read_csv(file_config.energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    if equalize_timesteps:
        plan = plan.equalize_timesteps()
        logger.debug(f"Equalized timesteps to {plan.time[0]}")

    plan = plan.add_energy_price(
        energy_price["time"].to_list(), energy_price["energy_price"].to_list()
    )
    plan = plan.add_grid_tariff(1.3e-4)

    optimizer_config = OptimizerConfig()

    # Get optimal initial state

    if optimizer_config.optimizer_type == "casadi":
        optimizer = CasadiOptimizer(
            plan,
            config=optimizer_config,
        )
    else:
        optimizer = GurobiOptimizer(
            plan,
            config=optimizer_config,
        )

    optimizer.build(ce_function_type="quadratic", alpha=optimizer_config.alpha)
    with suppress_stdout_stderr():
        global_solution = optimizer.solve()

    assert global_solution is not None
    initial_soe = [soe[0] for soe in global_solution.state_of_energy]
    eta_max = 0.95

    def charging_efficiency(p):
        return eta_max * (
            p - (optimizer_config.alpha) * p**2 / (2 * plan.max_charging_power)
        )

    env = Environment(
        plan, initial_soe, charging_efficiency, sigma=optimizer_config.energy_std_dev
    )
    looped_plan = plan.loop(days)

    charging_power = [[] for _ in range(plan.num_vehicles)]
    effective_charging_power = [[] for _ in range(plan.num_vehicles)]
    state_of_energy = [[initial_soe[vehicle]] for vehicle in range(plan.num_vehicles)]

    num_steps = len(plan.time) * days
    energy_cost = 0
    max_charging_power = 0
    k = 0
    policy = None
    current_soe = initial_soe

    logger.info("Running simulation")
    step_generator = range(num_steps) if debug else tqdm(range(num_steps))
    for i in step_generator:
        logger.debug(f"Step {i + 1}")
        if any(
            soe is not None and soe < threshold * cap
            for soe, threshold, cap in zip(
                current_soe, plan.soe_lb, plan.battery_capacity
            )
        ):
            logger.warning("  [orange1]Invalid state encountered -- stopping early")
            break

        # optimize and find policy
        if k == 0:
            logger.debug(
                f"  [light_sea_green]Optimizing the next {steps_until_reoptimization} steps"
            )
            optimizer_config.initial_soe = current_soe
            optimizer = GurobiOptimizer(env.plan, config=optimizer_config)
            optimizer.build(ce_function_type="quadratic", alpha=optimizer_config.alpha)
            with suppress_stdout_stderr():
                solution = optimizer.solve()
            if solution is None:
                logger.warning(
                    "  [orange1]Optimizer encountered infeasible problem -- stopping early"
                )
                break
            policy = policy_from_solution(solution, steps_until_reoptimization)
        logger.debug(
            f"  Current SoE: ({', '.join([f'{soe:.5f}' if soe is not None else '---' for soe in current_soe])[:-1]})"
        )
        logger.debug(f"  Policy: ({', '.join([f'{cp:.5f}' for cp in policy[k]])[:-1]})")

        # track energy cost and max charging power
        energy_cost += sum(
            cp * env.plan.time[0] * env.plan.energy_price[0] for cp in policy[k]
        )
        max_charging_power = max(max_charging_power, max(policy[k]))

        for vehicle, cp in enumerate(policy[k]):
            charging_power[vehicle].append(cp)
            effective_charging_power[vehicle].append(charging_efficiency(cp))

        # update state of energy
        current_soe = env.step(policy[k])
        for vehicle, soe in enumerate(current_soe):
            state_of_energy[vehicle].append(env.soe[vehicle])

        k = (k + 1) % steps_until_reoptimization
        logger.debug(
            "----------------------------------------------------------------------"
        )

    power_cost = 1.3e-4 * max_charging_power * days
    total_cost = energy_cost + power_cost

    total_cost_str = f"{total_cost:.3f} $"
    energy_cost_str = f"{energy_cost:.3f} $"
    power_cost_str = f"{power_cost:.3f} $"
    max_cost_string_length = max(
        map(len, [total_cost_str, energy_cost_str, power_cost_str])
    )
    logger.info(
        f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost_str))}{total_cost_str}"
    )
    logger.info(
        f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost_str))}{energy_cost_str}"
    )
    logger.info(
        f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost_str))}{power_cost_str}"
    )

    if len(charging_power[0]) < looped_plan.num_timesteps:
        looped_plan = looped_plan.truncate(len(charging_power[0]))

    solution = Solution(
        input_data=looped_plan,
        total_cost=total_cost,
        energy_cost=energy_cost,
        power_cost=power_cost,
        gap=0.0,
        max_charging_power_used=max_charging_power,
        charging_power=charging_power,
        effective_charging_power=effective_charging_power,
        state_of_energy=state_of_energy,
        lower_soe_envelope=state_of_energy,
    )

    with open(file_config.solution_file, "w") as f:
        f.write(solution.model_dump_json(indent=4))
    logger.info(f"Saved solution to [cyan3]{file_config.solution_file}")


@click.command()
# general options
@click.option("--debug", is_flag=True, default=False, help="print debug messages")
@click.option(
    "--steps-until-reoptimization",
    type=int,
    default=10,
    help="number of steps taken before reoptimizing policy",
)
@click.option(
    "--equalize-timesteps",
    is_flag=True,
    default=False,
    help="equalize timesteps of data",
)
@click.option("--days", type=int, default=10, help="number of days simulated")
@FileConfig.as_click_options
@OptimizerConfig.as_click_options
def main(
    debug: bool,
    steps_until_reoptimization: int,
    equalize_timesteps: bool,
    days: int,
    file_config: FileConfig,
    optimizer_config: OptimizerConfig,
):
    return run_main(
        debug=debug,
        steps_until_reoptimization=steps_until_reoptimization,
        equalize_timesteps=equalize_timesteps,
        days=days,
        file_config=file_config,
        optimizer_config=optimizer_config,
    )
