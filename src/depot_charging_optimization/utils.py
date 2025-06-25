import re
from typing import Iterable, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


def numpy_to_py(x):
    if isinstance(x, np.ndarray) or isinstance(x, list):
        return [numpy_to_py(v) for v in x]
    elif isinstance(x, (np.generic,)):
        return x.item()
    else:
        return x


def py_to_numpy(x):
    type_map = {
        int: np.int64,
        float: np.float32,
        bool: np.bool_,
    }
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return np.array([py_to_numpy(v) for v in x])
    elif type(x) in type_map.keys():
        return type_map[type(x)](x)
    else:
        return x


def list_start_string(values: Iterable, num: int) -> str:
    s = str(values[:num])
    if len(values) <= num:
        return s
    else:
        return s[:-1] + ", ...]"


def minimum_joint_chain_range(blocks: list, joints: list) -> int | float:
    assert len(blocks) == len(joints) + 1
    if len(blocks) == 1:
        return blocks[0]
    sub_chain_solution = minimum_joint_chain_range(blocks[1:], joints[1:])
    if joints[0] < blocks[0]:
        return max(blocks[0], blocks[0] - joints[0] + sub_chain_solution)
    else:
        return max(blocks[0], sub_chain_solution)


def find_continuos_blocks(values: Sequence[T]) -> list[tuple[int, int, T]]:
    continuous_blocks = []
    last_change = 0
    for v1, (i, v2) in zip(values, enumerate(values[1:])):
        if v1 != v2:
            continuous_blocks.append((last_change, i + 1, v1))
            last_change = i + 1
    continuous_blocks.append((last_change, len(values), values[-1]))
    if continuous_blocks[0][2] == continuous_blocks[-1][2]:
        continuous_blocks = continuous_blocks[1:-1] + [
            (continuous_blocks[-1][0], continuous_blocks[0][1], continuous_blocks[-1][2])
        ]
    return continuous_blocks


def partial_sums(iterable: Iterable) -> Iterable:
    total = 0
    for i in iterable:
        total += i
        yield total


def atoi(text: str) -> int | str:
    return int(text) if text.isdigit() else text


def group_vehicles_by_index(data: list[np.ndarray]) -> dict:
    grouped = {}
    for vehicle, indices in enumerate(data):
        for index in indices:
            if index not in grouped:
                grouped[index] = [vehicle]
            else:
                grouped[index].append(vehicle)
    return grouped


def natural_keys(text: str) -> list:
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def expand_values(time: Iterable[int], values: Iterable[T], granularity: int, interpolation: str = "same") -> list[T]:
    expanded_values = []
    current_time = 0
    current_value = 0
    eps = 1e-6
    for t, v in zip(time, values):
        assert t >= current_time
        assert t % granularity == 0
        num = (t - current_time) // granularity
        current_time = t
        if interpolation == "same":
            expanded_values += [v] * num
        elif interpolation == "split" and type(v) in [float, int]:
            ev = v / num
            if isinstance(v, int):
                ev = int(ev)
            else:
                ev = float(ev)
            assert abs(ev * num - v) < eps
            expanded_values += [ev] * num
        elif interpolation == "linear" and type(v) in [float, int]:
            unit = (v - current_value) / num
            if isinstance(v, int):
                unit = int(unit)
            else:
                unit = float(unit)
            expanded_values += [current_value + unit * (i + 1) for i in range(num)]
            current_value = v
        else:
            raise ValueError(f"Invalid interpolation type: {interpolation} with type {type(v)}")
    return expanded_values
