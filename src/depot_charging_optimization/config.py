from functools import wraps
from pathlib import Path
from typing import ClassVar, Self, get_origin

import click
import yaml
from pydantic import BaseModel, field_validator


class BaseConfig(BaseModel):
    function_argument_name: ClassVar[str] = "config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("config", Path("config/optimizer.yaml"))

    def __repr__(self):
        return "\n".join(
            [
                f"{field[0]}: [{field[1].annotation}] = {self.model_dump()[field[0]]} "
                for field in self.__class__.model_fields.items()
            ]
        )

    def __str__(self):
        return self.__repr__()

    @classmethod
    def load_from_dict(cls, config_dict: dict) -> Self:
        if config_dict["config_file"].exists():
            with open(config_dict["config_file"]) as f:
                config_file_dict = yaml.safe_load(f)
        else:
            config_file_dict = {}
        config = cls(**config_file_dict)  # type: ignore
        return config.model_copy(update=config_dict)

    @classmethod
    def as_click_options(cls, func):
        options = cls.model_fields
        config_file_argument_name, config_file = cls.default_config

        @wraps(func)
        def wrapper(*args, **kwargs):
            config = {k: v for k, v in kwargs.items() if v is not None and k in options}
            if config_file_argument_name in kwargs.keys():
                config["config_file"] = kwargs[config_file_argument_name] or config_file
            else:
                config["config_file"] = config_file
            return func(
                *args,
                **{cls.function_argument_name: config},
                **{k: v for k, v in kwargs.items() if k not in options and k != config_file_argument_name},
            )

        wrapper = click.option(f"--{config_file_argument_name.replace('_', '-')}", type=Path)(wrapper)

        for name, info in reversed(options.items()):
            if get_origin(info.annotation) is not None:
                pass
            else:
                wrapper = click.option(f"--{name.replace('_', '-')}", type=info.annotation)(wrapper)

        return wrapper


class OptimizerConfig(BaseConfig):
    function_argument_name: ClassVar[str] = "optimizer_config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("optimizer_config", Path("config/optimizer.yaml"))
    optimizer_type: str = "casadi"
    alpha: float = 0.0
    max_efficiency: float = 1.0
    bidirectional_charging: bool = False
    confidence_level: float = 0.0
    energy_std_dev: float = 0.0
    initial_soe: list[float | None] | None = None

    @field_validator("optimizer_type")
    @classmethod
    def valid_optimizer_type(cls, v):
        if v.lower() not in ["gurobi", "casadi"]:
            raise ValueError(f"unknown optimizer type [{v}], must be either gurobi or casadi")
        return v.lower()

    @field_validator("alpha", "confidence_level", "energy_std_dev")
    @classmethod
    def between_zero_and_one(cls, v):
        if not (0 <= v <= 1):
            raise ValueError(f"Value must be between zero and one, got {v}")
        return v


class ModelPredictiveControlConfig(BaseConfig):
    function_argument_name: ClassVar[str] = "mpc_config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("mpc_config", Path("config/mpc.yaml"))
    minutes_until_reoptimization: int = 60

    @field_validator("minutes_until_reoptimization")
    @classmethod
    def strictly_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Value must be strictly positive, got {v}")
        return v


class FileConfig(BaseConfig):
    function_argument_name: ClassVar[str] = "file_config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("file_config", Path("config/file.yaml"))
    data_files: list[Path] = []
    energy_price_file: Path = Path("data/energy_price.csv")
    grid_tariff_file: Path = Path("data/grid_tariff.csv")
    solution_file: Path = Path("outputs/solutions/solution.json")

    @field_validator("energy_price_file", "grid_tariff_file")
    @classmethod
    def file_exists(cls, v):
        assert v.exists()
        return v


class EnvironmentConfig(BaseConfig):
    function_argument_name: ClassVar[str] = "env_config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("env_config", Path("config/env.yaml"))
    env_energy_std_dev: float = 0.0
    num_days: int = 1
    charger_max_charging_power: float = 0.0
    charger_max_efficiency: float = 1.0
    charger_loss_coefficient: float = 0.0


class HeuristicConfig(BaseConfig):
    function_argument_name: ClassVar[str] = "heuristic_config_cli_arguments"
    default_config: ClassVar[tuple[str, Path]] = ("heuristic_config", Path("config/heuristic.yaml"))
    heuristic_type: str = "charge_on_arrival"
    max_charging_power_allowed: float = 0.0
