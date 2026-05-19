from pathlib import Path

from depot_charging_optimization.config import (
    EnvironmentConfig,
    FileConfig,
    HeuristicConfig,
    ModelPredictiveControlConfig,
    OptimizerConfig,
)
from depot_charging_optimization.scripts.mpc import run_main as _run_mpc
from depot_charging_optimization.scripts.optimize import run_main as _run_optimize
from depot_charging_optimization.scripts.simulate import run_main as _run_simulate


def run_simulate(
    file_config_path: Path = FileConfig.default_config[1],
    env_config_path: Path = EnvironmentConfig.default_config[1],
    heuristic_config_path: Path = HeuristicConfig.default_config[1],
    debug: bool = False,
):
    file_config = FileConfig.load_from_dict({"config_file": file_config_path})
    env_config = EnvironmentConfig.load_from_dict({"config_file": env_config_path})
    heuristic_config = HeuristicConfig.load_from_dict({"config_file": heuristic_config_path})
    return _run_simulate(debug, file_config, env_config, heuristic_config)


def run_optimize(
    file_config_path: Path = FileConfig.default_config[1],
    optimizer_config_path: Path = OptimizerConfig.default_config[1],
    env_config_path: Path = EnvironmentConfig.default_config[1],
    debug: bool = False,
):
    file_config = FileConfig.load_from_dict({"config_file": file_config_path})
    optimizer_config = OptimizerConfig.load_from_dict({"config_file": optimizer_config_path})
    env_config = EnvironmentConfig.load_from_dict({"config_file": env_config_path})
    return _run_optimize(debug, file_config, optimizer_config, env_config)


def run_mpc(
    file_config_path: Path = FileConfig.default_config[1],
    env_config_path: Path = EnvironmentConfig.default_config[1],
    mpc_config_path: Path = ModelPredictiveControlConfig.default_config[1],
    optimizer_config_path: Path = OptimizerConfig.default_config[1],
    debug: bool = False,
):
    file_config = FileConfig.load_from_dict({"config_file": file_config_path})
    env_config = EnvironmentConfig.load_from_dict({"config_file": env_config_path})
    mpc_config = ModelPredictiveControlConfig.load_from_dict({"config_file": mpc_config_path})
    optimizer_config = OptimizerConfig.load_from_dict({"config_file": optimizer_config_path})
    return _run_mpc(debug, file_config, env_config, mpc_config, optimizer_config)
