from typing import Protocol

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


def peak_shaving(env: Environment, max_power_ratio: float = 1.0, *args, **kwargs) -> list[float]:
    policy = []
    assert env.state is not None
    time_until_next_departure = []
    doubled_plan = env.plan.loop(2) # double plan to check departure on next day
    for vehicle in range(env.plan.num_vehicles):
        dt = 0
        for i, depot_charge in enumerate(doubled_plan.depot_charge[vehicle][env.timestep :]):
            if not depot_charge:
                break
            dt += env.time_delta[(env.timestep + i) % len(env.time_delta)]
        time_until_next_departure.append(dt)

    for i, (charge_possible, soe, cap) in enumerate(
        zip(env.state.in_depot, env.state.state_of_energy, env.state.battery_capacity)
    ):
        if charge_possible:
            chargeable_energy = cap - soe
            effective_charging_power = chargeable_energy / time_until_next_departure[i]
            try:
                power_required = env.charger.inverse_effective_charging_power(effective_charging_power)
                if power_required > env.charger.max_charging_power * max_power_ratio:
                    policy.append(env.charger.max_charging_power * max_power_ratio)
                else:
                    policy.append(power_required)
            except ValueError:
                policy.append(env.charger.max_charging_power * max_power_ratio)
        else:
            policy.append(0.0)
    return policy
