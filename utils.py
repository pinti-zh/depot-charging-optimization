import re

import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import GRB
from rich import print


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def print_model_summary(model, verbosity):
    if verbosity >= 1:
        model.setParam("OutputFlag", 1)
        print("[bold]" + "=" * 100)
        print(" " * 40 + "[bold]OPTIMIZATION PROBLEM" + 40 * " ")
        print("[bold]" + "=" * 100)
        model.printStats()
        if verbosity >= 2:
            print("[bold]" + "-" * 100)
            print("[bold]Constraints:")
            max_name_len = max(len(constraint.ConstrName) for constraint in model.getConstrs())
            max_constraint_len = max(len(str(model.getRow(constraint))) for constraint in model.getConstrs())
            for constraint in sorted(model.getConstrs(), key=lambda x: natural_keys(x.ConstrName)):
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
                print(f"  {name_buffer}[bright_blue italic]{constraint.ConstrName}[/bright_blue italic]: ", end="")
                constraint_buffer = (max_constraint_len - len(str(model.getRow(constraint)))) * " "
                print(f"{constraint_buffer}{model.getRow(constraint)} {sense} {constraint.RHS}")

            print("\n[bold]Variable Bounds:")
            max_variable_len = max(len(variable.VarName) for variable in model.getVars())
            max_lower_bound_len = max(len(str(variable.LB)) for variable in model.getVars())
            for variable in sorted(model.getVars(), key=lambda x: natural_keys(x.VarName)):
                variable_buffer = (max_variable_len - len(variable.VarName)) * " "
                lb_buffer = (max_lower_bound_len - len(str(variable.LB))) * " "
                print(f"  {lb_buffer}{variable.LB} [bold]<=[/bold] ", end="")
                print(f"[bright_blue italic]{variable_buffer}{variable.VarName}[/bright_blue italic] ", end="")
                print(f"[bold]<=[/bold] {variable.UB}")

        print("[bold]" + "=" * 100)
        model.setParam("OutputFlag", 0)


def plot_result(power, state_of_energy, df, state_of_energy_lower_bound, state_of_energy_upper_bound):
    num_timesteps = len(df)

    sns.set_style("darkgrid")
    _, axes = plt.subplots(2, height_ratios=[2, 1], figsize=(12, 8))

    # y = 0 line
    axes[0].plot([0, num_timesteps], [0, 0], c="black", linestyle="dashed")

    # state of energy
    axes[0].plot(
        range(num_timesteps + 1),
        [state_of_energy[t].X for t in range(num_timesteps + 1)],
        c="navy",
        label="SoE",
    )
    # soe lower bound
    axes[0].plot(
        range(num_timesteps + 1), [state_of_energy_lower_bound] * (num_timesteps + 1), linestyle="dashed", c="navy"
    )
    # soe upper bound
    axes[0].plot(
        range(num_timesteps + 1), [state_of_energy_upper_bound] * (num_timesteps + 1), linestyle="dashed", c="navy"
    )

    # power
    power_values = [power[t].X for t in range(num_timesteps)]
    power_values = [val for val in power_values for _ in (0, 1)]  # duplicate power values
    time_steps = [0] + [t for t in range(1, num_timesteps) for _ in (0, 1)] + [num_timesteps]
    positive_power_values = list(map(lambda x: max(0, x), power_values))
    negative_power_values = list(map(lambda x: min(0, x), power_values))
    axes[0].plot(
        time_steps, negative_power_values, c="firebrick", linestyle="dashed", label="power consumption"
    )  # power consumption
    axes[0].plot(time_steps, positive_power_values, c="firebrick", label="charging power")  # charging power

    # axis 0 config
    axes[0].set_ylim(-max(df["EnergyDemand"]) * 1.1, 100)
    axes[0].legend()
    axes[0].set_title("SoE and Power")

    # energy price
    energy_price_values = [val for val in df["EnergyPrice"] for _ in (0, 1)]
    axes[1].plot(time_steps, energy_price_values, c="forestgreen")

    # axis 1 config
    axes[1].set_ylim(0, max(df["EnergyPrice"]) * 1.1)
    axes[1].set_title("Energy Price")

    plt.tight_layout()
    plt.show()
