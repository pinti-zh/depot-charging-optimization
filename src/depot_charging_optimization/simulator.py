from typing import Any, Protocol

from depot_charging_optimization.environment import Environment


class HeuristicFunction(Protocol):
    def __call__(self, env: Environment) -> list[float]: ...


def charge_on_arrival(env: Environment) -> list[float]:
    policy = []
    dt = env.time_delta[env.timestep]
    assert env.state is not None
    for charge_possible, soe, cap in zip(env.state.in_depot, env.state.state_of_energy, env.state.battery_capacity):
        if charge_possible:
            chargeable_energy = cap - soe
            effective_charging_power = chargeable_energy / dt
            if effective_charging_power > env.charger.max_possible_effective_charging_power():
                policy.append(env.charger.max_charging_power)
            else:
                policy.append(env.charger.inverse_effective_charging_power(effective_charging_power))
        else:
            policy.append(0.0)
    return policy
