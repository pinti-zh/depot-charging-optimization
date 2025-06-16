import re
from typing import Iterable, TypeVar

import numpy as np

T = TypeVar("T")


def list_start_string(values: Iterable, num: int) -> str:
    s = str(values[:num])
    if len(values) <= num:
        return s
    else:
        return s[:-1] + ", ...]"


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
