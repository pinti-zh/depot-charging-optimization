import json
from copy import deepcopy

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
@click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
def mcp(data_files, energy_price_file):
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

    # quick test
    plan_copy = deepcopy(plan)
    test_soe = None
    for i in range(100):
        optimizer = GurobiOptimizer(plan_copy, bidirectional_charging=False, initial_soe=test_soe)
        optimizer.build(ce_function_type="quadratic", alpha=0.8)
        with suppress_stdout_stderr():
            solution = optimizer.solve()
        logger.info(f"Run {i + 1}, Total Cost: {solution.total_cost}")
        plan_copy = plan_copy.rotate()
        test_soe = [soe[0] for soe in solution.state_of_energy]

    return

    # Get optimal initial state
    initial_soe = [bc * 0.5 for bc in plan.battery_capacity]
    optimizer = GurobiOptimizer(plan, bidirectional_charging=False, initial_soe=initial_soe)
    optimizer.build(ce_function_type="quadratic", alpha=0.8)
    with suppress_stdout_stderr():
        global_solution = optimizer.solve()

    initial_soe = [soe[0] for soe in global_solution.state_of_energy]
    env = Environment(initial_soe)

    # history
    time = [0]
    state_history = [[soe] for soe in initial_soe]

    num_steps = len(plan.time)
    energy_cost = 0
    max_charging_power = 0
    current_soe = initial_soe
    for i in range(num_steps):
        logger.info(f"Step {i + 1}")
        logger.info(f"Current SoE: {current_soe}")
        logger.info(f"Global SoE:  {[soe[i] for soe in global_solution.state_of_energy]}")

        # optimize and find policy
        optimizer = GurobiOptimizer(plan, bidirectional_charging=False, initial_soe=current_soe)
        optimizer.build(ce_function_type="quadratic", alpha=0.8)
        with suppress_stdout_stderr():
            solution = optimizer.solve()
        if solution is None:
            break
        policy = policy_from_solution(solution)
        logger.info(f"Global Policy: {[cp[i] for cp in global_solution.charging_power]}")
        logger.info(f"Local Policy: {policy}")

        # track energy cost and max charging power
        energy_cost += sum(cp * plan.time[0] * plan.energy_price[0] for cp in policy)
        max_charging_power = max(max_charging_power, max(policy))

        # update state of energy
        current_soe = env.step(plan, policy)
        for i, soe in enumerate(current_soe):
            state_history[i].append(soe)
        time.append(time[-1] + plan.time[0])

        plan = plan.rotate()

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
        plt.plot(time, [v / 3.6e6 if v is not None else 0.0 for v in trace])
    plt.show()
