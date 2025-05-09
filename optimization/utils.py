import re

import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import GRB
from matplotlib.patches import Rectangle
from rich import print


def print_params(data):
    print("[bold]" + "=" * 100)
    print("[bold]Parameters:")
    print(f"  numTimeSteps: {data['numTimeSteps']}")
    print(f"  timeStepDuration: {data['timeStepDuration']}")
    print(f"  powerGridTariff: {data['powerGridTariff']}")
    print(f"  maxChargingPower: {data['maxChargingPower']}")
    print(f"  stateOfEnergyLowerBound: {data['stateOfEnergyLowerBound']}")
    print(f"  stateOfEnergyUpperBound: {data['stateOfEnergyUpperBound']}")
    print(f"  energyPrice: {len(data['energyPrice'])} Prices")
    print(f"  energyDemand: {[demand['value'] for demand in data['energyDemand']]}")
    print("[bold]" + "=" * 100)


def plot_solution(opt_model):
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, height_ratios=[2, 1], figsize=(12, 8))
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    fig.suptitle("Optimization Result", fontsize=16)
    axes[0].plot(range(opt_model.data["numTimeSteps"] + 1), [soe.X for soe in opt_model.state_of_energy], color="navy")
    axes[0].plot(
        range(opt_model.data["numTimeSteps"] + 1),
        [opt_model.data["stateOfEnergyLowerBound"]] * (opt_model.data["numTimeSteps"] + 1),
        color="navy",
        linestyle="dashed",
        label="SoE Lower Bound",
    )
    axes[0].plot(
        range(opt_model.data["numTimeSteps"] + 1),
        [opt_model.data["stateOfEnergyUpperBound"]] * (opt_model.data["numTimeSteps"] + 1),
        color="navy",
        linestyle="dashed",
        label="SoE Upper Bound",
    )
    axes[0].bar(
        [i + 0.5 for i in opt_model.charging_indices],
        [cp.X for cp in opt_model.charging_power],
        color="firebrick",
        width=1.0,
        label="Charging Power",
    )
    for energy_demand in opt_model.data["energyDemand"]:
        axes[0].add_patch(
            Rectangle(
                (energy_demand["start"], 0),
                energy_demand["end"] - energy_demand["start"],
                100,
                color="firebrick",
                alpha=0.2,
            )
        )
    axes[0].legend()
    axes[0].set_ylim(0, 100)
    axes[0].set_xlim(0, opt_model.data["numTimeSteps"])
    energy_time_steps = (
        [0] + [i for i in range(1, opt_model.data["numTimeSteps"]) for _ in (0, 1)] + [opt_model.data["numTimeSteps"]]
    )
    energy_price_values = [val for val in opt_model.data["energyPrice"] for _ in (0, 1)]
    axes[1].plot(energy_time_steps, energy_price_values, color="forestgreen", label="Energy Price")
    axes[1].set_xlim(0, opt_model.data["numTimeSteps"])
    axes[1].legend()
    plt.show()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def print_linear_constraints(model):
    constraints = model.getConstrs()
    print("[bold]Linear Constraints:")
    max_name_len = max(len(constraint.ConstrName) for constraint in constraints)
    max_constraint_len = max(len(str(model.getRow(constraint))) for constraint in constraints)
    for constraint in sorted(constraints, key=lambda x: natural_keys(x.ConstrName)):
        match constraint.Sense:
            case GRB.LESS_EQUAL:
                sense = "[bold]<=[/bold]"
            case GRB.EQUAL:
                sense = "[bold]==[/bold]"
            case GRB.GREATER_EQUAL:
                sense = "[bold]>=[/bold]"
            case _:
                sense = "???"
        name_buffer = (max_name_len - len(constraint.ConstrName)) * " "
        print(
            f"  {name_buffer}[bright_blue italic]{constraint.ConstrName}[/bright_blue italic]: ",
            end="",
        )
        constraint_buffer = (max_constraint_len - len(str(model.getRow(constraint)))) * " "
        print(f"{constraint_buffer}{model.getRow(constraint)} {sense} {constraint.RHS}")


def print_general_constraints(model):
    constraints = model.getGenConstrs()
    print("[bold]General Constraints:")
    max_constraint_len = max(len(constraint.genConstrName) for constraint in constraints)
    for constraint in constraints:
        name_buffer = (max_constraint_len - len(constraint.genConstrName)) * " "
        print(f"  {name_buffer}[bright_blue italic]{constraint.genConstrName}")


def print_variables(model):
    print("[bold]Variable Bounds:")
    max_variable_len = max(len(variable.VarName) for variable in model.getVars())
    max_lower_bound_len = max(len(str(variable.LB)) for variable in model.getVars())
    for variable in sorted(model.getVars(), key=lambda x: natural_keys(x.VarName)):
        variable_buffer = (max_variable_len - len(variable.VarName)) * " "
        lb_buffer = (max_lower_bound_len - len(str(variable.LB))) * " "
        print(f"  {lb_buffer}{variable.LB} [bold]<=[/bold] ", end="")
        print(f"[bright_blue italic]{variable_buffer}{variable.VarName}[/bright_blue italic] ", end="")
        print(f"[bold]<=[/bold] {variable.UB}")


def print_model_summary(opt_model, verbosity):
    if verbosity >= 1:
        opt_model.model.setParam("OutputFlag", 1)
        print("[bold]" + "=" * 100)
        print(" " * 40 + "[bold]OPTIMIZATION PROBLEM" + 40 * " ")
        print("[bold]" + "=" * 100)
        opt_model.model.printStats()
        if verbosity >= 2:
            print("[bold]" + "-" * 100)

            if len(opt_model.model.getConstrs()):
                print_linear_constraints(opt_model.model)
                print()
            if len(opt_model.model.getGenConstrs()):
                print_general_constraints(opt_model.model)
                print()
            if len(opt_model.model.getVars()):
                print_variables(opt_model.model)

        print("[bold]" + "=" * 100)
        opt_model.model.setParam("OutputFlag", 0)


def print_solution(opt_model, verbosity):
    if verbosity >= 2:
        soe_strings = list(map(lambda x: f"{x.X:.2f}", opt_model.state_of_energy))
        cp_strings = []
        cp_index = 0
        for i in range(opt_model.data["numTimeSteps"]):
            if i in opt_model.charging_indices:
                cp_strings.append(f"{opt_model.charging_power[cp_index].X:.2f}")
                cp_index += 1
            else:
                cp_strings.append(" ")
        max_string_len = max(max(map(len, soe_strings)), max(map(len, cp_strings)))
        max_string_len = max(max(map(len, soe_strings)), max(map(len, cp_strings)))
        for soe_string in soe_strings:
            buffer = " " * (max_string_len - len(soe_string))
            print(f"[bold blue]{buffer}{soe_string}{' ' * max_string_len}", end="")
        print()
        for cp_string in cp_strings:
            buffer = " " * (max_string_len - len(cp_string))
            print(f"[bold red]{' ' * max_string_len}{buffer}{cp_string}", end="")
        print(f"[bold red]{' ' * max_string_len} -> {sum([cp.X for cp in opt_model.charging_power]):.2f}")
        print("[bold]" + "-" * 100)
    if verbosity >= 1:
        grid_cost = opt_model.max_charging_power.X * opt_model.data["powerGridTariff"]
        energy_cost = sum(
            opt_model.data["energyPrice"][i] * cp.X * opt_model.data["timeStepDuration"]
            for i, cp in zip(opt_model.charging_indices, opt_model.charging_power)
        )
        print(f"Energy Cost:        [bold green]{energy_cost:.2f}$")
        print(f"Grid Cost:        + [bold green]{grid_cost:.2f}$")
        print("-------------------------")
        print(f"Total Optimal Cost: [bold green]{opt_model.model.ObjVal:.2f}$")
    else:
        print(opt_model.solution)
