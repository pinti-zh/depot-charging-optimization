from functools import wraps
from pathlib import Path
from typing import Any, get_origin

import click
import yaml  # type: ignore
from pydantic import BaseModel, field_validator, model_validator


class BaseConfig(BaseModel):
    config_file: Path | None = None
    _function_argument_name: str = "config"

    @model_validator(mode="before")
    @classmethod
    def load_from_config(cls, data: Any) -> Any:
        if isinstance(data, cls):
            return data  # already model instance

        data = data or {}  # make sure its mutable

        config_file = data.get("config_file")

        if config_file is not None and Path(config_file).exists():
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            return {**config_data, **data}

        return data

    def __repr__(self):
        return "\n".join(
            [f"{field[0]}: [{field[1].annotation}] = {self.dict()[field[0]]} " for field in self.model_fields.items()]
        )

    def __str__(self):
        return self.__repr__()

    @classmethod
    def as_click_options(cls, func):
        options = cls.model_fields

        @wraps(func)
        def wrapper(*args, **kwargs):
            config = cls(**{k: v for k, v in kwargs.items() if v is not None})
            return func(
                *args,
                **{config._function_argument_name: config},
                **{k: v for k, v in kwargs.items() if k not in cls.model_fields},
            )

        for name, info in reversed(options.items()):
            if get_origin(info.annotation) is not None:
                pass
            else:
                wrapper = click.option(f"--{name.replace('_', '-')}", type=info.annotation)(wrapper)

        return wrapper


class OptimizerConfig(BaseConfig):
    optimizer_type: str = "gurobi"
    ce_function_type: str = "one"
    alpha: float = 0.0
    max_efficiency: float = 1.0
    bidirectional_charging: bool = False
    confidence_level: float = 0.0
    energy_std_dev: float = 0.0
    initial_soe: list[float | None] | None = None
    config_file: Path | None = Path("config/optimizer.yaml")
    _function_argument_name: str = "optimizer_config"

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


class FileConfig(BaseConfig):
    data_files: list[Path] = []
    energy_price_file: Path = Path("data/energy_price.csv")
    grid_tariff_file: Path = Path("data/grid_tariff.csv")
    solution_file: Path = Path("outputs/solutions/solution.json")
    config_file: Path | None = Path("config/file.yaml")
    _function_argument_name: str = "file_config"

    @field_validator("energy_price_file", "grid_tariff_file")
    def file_exists(cls, v):
        assert v.exists()
        return v
