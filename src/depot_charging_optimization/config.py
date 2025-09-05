from typing import get_origin

import click
from pydantic import BaseModel, field_validator


class BaseConfig(BaseModel):
    def __repr__(self):
        return "\n".join(
            [f"{field[0]}: [{field[1].annotation}] = {self.dict()[field[0]]} " for field in self.model_fields.items()]
        )

    def __str__(self):
        return self.__repr__()

    @classmethod
    def config_options(cls, func):
        @cls._generate_click_options()
        def wrapper(*args, **kwargs):
            filtered_kwargs = dict((key, value) for key, value in kwargs.items() if value is not None)
            filtered_kwargs = dict(
                (key, list(value)) for key, value in filtered_kwargs.items() if isinstance(value, tuple)
            )
            config = OptimizerConfig(**filtered_kwargs)
            return func(config)

        return wrapper

    @classmethod
    def _generate_click_options(cls):
        option_list = [(field[0], field[1].annotation) for field in cls.model_fields.items()]

        def decorator(f):
            for name, field_type in option_list:
                param_name = f"--{name.replace('_', '-')}"
                if get_origin(field_type) is not None:  # no cli argument for complex types
                    pass
                elif field_type is bool:
                    f = click.option(param_name, is_flag=True)(f)
                else:
                    f = click.option(param_name, type=field_type)(f)
            return f

        return decorator


class OptimizerConfig(BaseConfig):
    optimizer_type: str = "gurobi"
    ce_function_type: str = "one"
    alpha: float = 1.0
    bidirectional_charging: bool = False
    confidence_level: float = 0.0
    energy_std_dev: float = 0.0
    initial_soe: list[float | None] | None = None

    @field_validator("optimizer_type")
    def valid_optimizer_type(cls, v):
        if v.lower() not in ["gurobi", "casadi"]:
            raise ValueError(f"unknown optimizer type [{v}], must be either gurobi or casadi")
        return v.lower()

    @field_validator("ce_function_type")
    def valid_ce_function_type(cls, v):
        if v.lower() not in ["one", "constant", "quadratic"]:
            raise ValueError(
                f"unknown charging efficiency function type [{v}], must be either one, contant, or quadratic"
            )
        return v.lower()

    @field_validator("alpha", "confidence_level", "energy_std_dev")
    def between_zero_and_one(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Value must be between zero and one, got {v}")
        return v


# optimize arguments

# @click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
# @click.option("--bidirectional_charging", "-bc", is_flag=True, default=False, help="allow no bidirectional charging")
# @click.option("--use_casadi", "-uc", is_flag=True, default=False, help="use casadi instead of gurobi")
# @click.option("--confidence_level", "-cl", type=float, default=0.0, help="confidence level for stochastic robustness")
# @click.option(
#     "--energy_std_dev", "-esd", type=float, default=0.0, help="energy standard deviation for stochastic robustness"
# )
# @click.option(
#     "--ce_function", "-cef", type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False), default="one"
# )
# -> OptimizerConfig

# @click.argument("data_files", type=str, nargs=-1)
# @click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
# @click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")
# @click.option("--time_limit", "-tl", type=int, default=5, help="solver time limit in seconds")
# @click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")

# mcp arguments
# @click.command()
# @click.argument("data_files", type=str, nargs=-1)
# @click.option("--energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
# @click.option(
#     "--steps_until_reoptimization",
#     "-reop",
#     type=int,
#     default=10,
#     help="number of steps taken before reoptimizing policy",
# )
# @click.option("--days", type=int, default=10, help="number of days simulated")
# @click.option("--alpha", "-a", type=float, default=1.0, help="charging efficiency")
# @click.option("--sigma", "-s", type=float, default=0.0, help="standard deviation of energy demand")
# @click.option("--equalize_timesteps", "-eqt", is_flag=True, default=False, help="equalize timesteps of data")
# @click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")

# simulate arguments
# @click.command()
# @click.argument("data_files", type=str, nargs=-1)
# @click.option("energy_price_file", "-epf", type=str, default="data/energy_price.csv", help="energy price file")
# @click.option(
#     "--ce_function", "-cef", type=click.Choice(["constant", "quadratic", "one"], case_sensitive=False), default="one"
# )
# @click.option(
#     "--simulation_algorithm",
#     "-sa",
#     type=click.Choice(["greedy", "peak_shaving"], case_sensitive=False),
#     default="greedy",
# )
# @click.option("--alpha", "-a", type=float, default=1.0, help="constant for charging efficiency function")
# @click.option("--solution_file", "-sf", type=str, default="outputs/solutions/solution.json", help="solution file")
# @click.option("--debug", "-d", is_flag=True, default=False, help="print debug messages")
# @click.option(
#     "--max_charging_power",
#     "-mcp",
#     type=float,
#     default=1.0,
#     help="maximum charging power for peak shaving (between 0 and 1)",
# )
