import json

import click
import pandas as pd
from tqdm import tqdm

from depot_charging_optimization.controller import policy_from_solution
from depot_charging_optimization.core import GurobiOptimizer
from depot_charging_optimization.data_models import Input, Solution
from depot_charging_optimization.environment import Environment
from depot_charging_optimization.logging import get_logger, suppress_stdout_stderr


@click.command()
@click.argument("data_files", type=str, nargs=-1)
@click.option("--energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
@click.option(
    "--steps_until_reoptimization",
    "-reop",
    type=int,
    default=10,
    help="number of steps taken before reoptimizing policy",
)
@click.option("--days", type=int, default=10, help="number of days simulated")
@click.option("--alpha", "-a", type=float, default=1.0, help="charging efficiency")
@click.option("--sigma", "-s", type=float, default=0.0, help="standard deviation of energy demand")
@click.option("--equalize_timesteps", "-eqt", is_flag=True, default=False, help="equalize timesteps of data")
@click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")
def mcp(data_files, energy_price_file, steps_until_reoptimization, days, alpha, sigma, equalize_timesteps, debug):
    if debug:
        logger = get_logger(name="mcp", level="debug")
    else:
        logger = get_logger(name="mcp", level="info")

    input_data = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            input_data.append(Input.model_validate(json.load(f)))
    plan = Input.combine(input_data)

    energy_price = pd.read_csv(energy_price_file)
    energy_price["energy_price"] /= 3.6e6

    if equalize_timesteps:
        plan = plan.equalize_timesteps()
        logger.debug(f"Equalized timesteps to {plan.time[0]}")

    plan = plan.add_energy_price(energy_price["time"].to_list(), energy_price["energy_price"].to_list())
    plan = plan.add_grid_tariff(1.2e-4)

    # Get optimal initial state
    optimizer = GurobiOptimizer(plan, bidirectional_charging=False)
    optimizer.build(ce_function_type="quadratic", alpha=alpha)
    with suppress_stdout_stderr():
        global_solution = optimizer.solve()

    initial_soe = [soe[0] for soe in global_solution.state_of_energy]

    def charging_efficiency(p):
        n = p / plan.max_charging_power
        e = n - (1 - alpha) * n**2 / 2
        return e * plan.max_charging_power

    env = Environment(plan, initial_soe, charging_efficiency, sigma=sigma)
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

    step_generator = range(num_steps) if debug else tqdm(range(num_steps))
    logger.info("Running simulation")
    for i in step_generator:
        logger.debug(f"Step {i + 1}")

        # optimize and find policy
        if k == 0:
            logger.debug(f"  [light_sea_green]Optimizing the next {steps_until_reoptimization} steps")
            optimizer = GurobiOptimizer(env.plan, bidirectional_charging=False, initial_soe=current_soe)
            optimizer.build(ce_function_type="quadratic", alpha=0.8)
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

    power_cost = 1.2e-4 * max_charging_power * days
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
        gap=0.0,
        max_charging_power_used=max_charging_power,
        charging_power=charging_power,
        effective_charging_power=effective_charging_power,
        state_of_energy=state_of_energy,
    )
    solution_file = "outputs/solutions/solution.json"
    with open(solution_file, "w") as f:
        f.write(solution.model_dump_json(indent=4))
    logger.info(f"Saved solution to [cyan3]{solution_file}")
