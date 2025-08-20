from depot_charging_optimization.data_models import Solution


def policy_from_solution(solution: Solution, num_steps: int) -> list[list[float | None]]:
    return [[cp[i] for cp in solution.charging_power] for i in range(num_steps)]
