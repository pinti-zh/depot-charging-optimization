import json

import click
import matplotlib.pyplot as plt
import pandas as pd

from depot_charging_optimization.controller import policy_from_solution
from depot_charging_optimization.core import GurobiOptimizer
from depot_charging_optimization.data_models import Input
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
def mcp(data_files, energy_price_file, steps_until_reoptimization, days, alpha):
    logger = get_logger("mcp")

    input_data = []
    for data_file in data_files:
        with open(data_file, "r") as f:
            input_data.append(Input.model_validate(json.load(f)))
    plan = Input.combine(input_data)

    energy_price = pd.read_csv(energy_price_file)
    energy_price["energy_price"] /= 3.6e6

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

    env = Environment(plan, initial_soe, charging_efficiency)

    # history
    time = [0]
    state_history = [[soe] for soe in initial_soe]

    num_steps = len(plan.time) * days
    energy_cost = 0
    max_charging_power = 0
    k = 0
    policy = None
    current_soe = initial_soe
    for i in range(num_steps):
        logger.info(f"Step {i + 1}")
        logger.info(
            f"  Current SoE: ({', '.join([f'{soe:.5f}' if soe is not None else '---' for soe in current_soe])[:-1]})"
        )

        # optimize and find policy
        if k == 0:
            logger.info(f"  [orange1]Optimizing the next {steps_until_reoptimization} steps")
            optimizer = GurobiOptimizer(env.plan, bidirectional_charging=False, initial_soe=current_soe)
            optimizer.build(ce_function_type="quadratic", alpha=0.8)
            with suppress_stdout_stderr():
                solution = optimizer.solve()
            if solution is None:
                break
            policy = policy_from_solution(solution, steps_until_reoptimization)
        logger.info(f"  Policy: ({', '.join([f'{cp:.5f}' for cp in policy[k]])[:-1]})")

        # track energy cost and max charging power
        energy_cost += sum(cp * env.plan.time[0] * env.plan.energy_price[0] for cp in policy[k])
        max_charging_power = max(max_charging_power, max(policy[k]))

        # update state of energy
        current_soe = env.step(policy[k])
        for i, soe in enumerate(current_soe):
            state_history[i].append(soe)
        time.append(time[-1] + env.plan.time[0])
        k = (k + 1) % steps_until_reoptimization
        logger.info("----------------------------------------------------------------------")

    power_cost = 1.2e-4 * max_charging_power
    total_cost = energy_cost + power_cost
    total_cost = f"{total_cost:.3f} $"
    energy_cost = f"{energy_cost:.3f} $"
    power_cost = f"{power_cost:.3f} $"
    max_cost_string_length = max(map(len, [total_cost, energy_cost, power_cost]))
    logger.info(f"Total cost of solution:   {' ' * (max_cost_string_length - len(total_cost))}{total_cost}")
    logger.info(f"Energy cost of solution:  {' ' * (max_cost_string_length - len(energy_cost))}{energy_cost}")
    logger.info(f"Power cost of solution:   {' ' * (max_cost_string_length - len(power_cost))}{power_cost}")

    for trace in state_history:
        plt.plot(time, [v / 3.6e6 if v is not None else None for v in trace])
    plt.show()
