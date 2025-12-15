from datetime import datetime, timedelta

import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pydantic import BaseModel

from depot_charging_optimization.data_models import Solution


class PlotData(BaseModel):
    x: list[str]
    y: list[list[float]] | list[list[int]]
    width: list[float] | list[int] | None = None


def state_of_energy_trajectories_figure(solution: Solution, names: list[str], selection_mask: list[bool]) -> go.Figure:
    assert solution.input_data.num_vehicles == len(names) == len(selection_mask)
    plot_data = get_state_of_energy_plot_data(solution)

    fig = go.Figure()
    for i, show in enumerate(selection_mask):
        if not show:
            continue
        color = color_wheel(i)
        fig.add_trace(
            go.Scatter(
                x=plot_data.x,
                y=plot_data.y[i],  # type: ignore
                mode="lines",
                marker=dict(color=color),
                line=dict(color=color),
                name=names[i],
            )
        )
    update_layout(fig, yaxis_title="State of Energy [kWh]")
    return fig


def cumulative_charging_power_figure(solution: Solution, names: list[str], selection_mask: list[bool]) -> go.Figure:
    assert solution.input_data.num_vehicles == len(names) == len(selection_mask)
    plot_data_bars = get_charging_power_plot_data(solution)

    fig = go.Figure()
    for i, show in enumerate(selection_mask):
        if not show:
            continue
        color = color_wheel(i)
        fig.add_trace(
            go.Bar(
                x=plot_data_bars.x,
                y=plot_data_bars.y[i],  # type: ignore
                width=plot_data_bars.width,  # type: ignore
                marker_color=color,
                marker=dict(line=dict(width=0)),
                opacity=0.8,
                name=names[i],
            )
        )
    plot_data_line = get_cumulative_charging_power_plot_data(solution)
    fig.add_trace(
        go.Scatter(
            x=plot_data_line.x,
            y=plot_data_line.y[0],  # type: ignore
            mode="lines",
            marker=dict(color="white"),
            line=dict(color="white"),
            name="total",
        )
    )
    fig.update_layout(
        barmode="relative",
    )
    update_layout(fig, yaxis_title="Charging Power [kW]")
    return fig


def energy_price_figure(solution: Solution) -> go.Figure:
    plot_data = get_energy_price_plot_data(solution)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_data.x,
            y=plot_data.y[0],  # type: ignore
            mode="lines",
        )
    )
    fig.update_layout(
        barmode="stack",
    )
    update_layout(fig, yaxis_title="Energy Price [CHF/kWh]")
    return fig


def detail_figure(solution: Solution, index: int, show_lower_envelope: bool = True) -> go.Figure:
    assert index < solution.input_data.num_vehicles

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color = color_wheel(index)

    if show_lower_envelope:
        lower_envelope_plot_data = get_lower_envelope_plot_data(solution)
        fig.add_trace(
            go.Scatter(
                x=lower_envelope_plot_data.x,
                y=lower_envelope_plot_data.y[index],  # type: ignore
                mode="lines",
                marker=dict(color=color),
                line=dict(dash="dot", color=color),
            )
        )

    # state of energy
    state_of_energy_plot_data = get_state_of_energy_plot_data(solution)
    fig.add_trace(
        go.Scatter(
            x=state_of_energy_plot_data.x,
            y=state_of_energy_plot_data.y[index],  # type: ignore
            mode="lines",
            marker=dict(color=color),
            line=dict(color=color),
        )
    )
    for bound in [
        0.0,
        solution.input_data.battery_capacity[index],
    ]:
        fig.add_trace(
            go.Scatter(
                x=state_of_energy_plot_data.x,
                y=[bound / 3.6e6 for _ in state_of_energy_plot_data.x],  # type: ignore
                mode="lines",
                marker=dict(color=color),
                line=dict(color=color, dash="dash"),
            )
        )

    # charging power
    charging_power_plot_data = get_charging_power_plot_data(solution)
    fig.add_trace(
        go.Bar(
            x=charging_power_plot_data.x,
            y=charging_power_plot_data.y[index],  # type: ignore
            width=charging_power_plot_data.width,  # type: ignore
            marker_color=color,
            marker=dict(line=dict(width=0)),
            opacity=0.4,
        ),
        secondary_y=True,
    )

    # depot charging intervals
    bands = get_depot_charge_bands(solution, index, color, 0.2)
    fig.update_layout(
        shapes=bands,
        yaxis2_title="Charging Power [kW]",
    )
    update_layout(fig, yaxis_title="State of Energy [kWh]")
    return fig


def input_data_figure(solution: Solution, index: int) -> go.Figure:
    assert index < solution.input_data.num_vehicles
    fig = go.Figure()

    # energy demands
    plot_data = get_energy_demand_plot_data(solution)

    fig.add_trace(
        go.Bar(
            x=plot_data.x,
            y=plot_data.y[index],  # type: ignore
            width=plot_data.width,  # type: ignore
            marker_color="#B82E2E",
            marker=dict(line=dict(width=0)),
        ),
    )

    # depot charging intervals
    bands = get_depot_charge_bands(solution, index, "#109618", 0.2)
    fig.update_layout(
        shapes=bands,
    )
    update_layout(fig, yaxis_title="Power Demand [kW]")

    return fig


def get_state_of_energy_plot_data(solution: Solution) -> PlotData:
    x = convert_seconds_to_time([0] + solution.input_data.time)
    y = [[soe / 3.6e6 for soe in soe_list] for soe_list in solution.state_of_energy]
    return PlotData(x=x, y=y)


def get_lower_envelope_plot_data(solution: Solution) -> PlotData:
    x = convert_seconds_to_time([0] + solution.input_data.time)
    y = [[soe / 3.6e6 for soe in soe_list] for soe_list in solution.lower_soe_envelope]
    return PlotData(x=x, y=y)


def get_charging_power_plot_data(solution: Solution) -> PlotData:
    seconds = [solution.input_data.time[0] // 2]
    width = [solution.input_data.time[0]]
    for t1, t2 in zip(solution.input_data.time[:-1], solution.input_data.time[1:]):
        width.append(t2 - t1)
        seconds.append((t1 + t2) // 2)
    x = convert_seconds_to_time(seconds)
    y = [[cp / 1000 for cp in cp_list] for cp_list in solution.charging_power]
    width = [w * 1000 for w in width]
    return PlotData(x=x, y=y, width=width)


def get_cumulative_charging_power_plot_data(solution: Solution) -> PlotData:
    x = convert_seconds_to_time(convert_to_interval(solution.input_data.time))
    cum_cp = [
        sum(solution.charging_power[i][j] for i in range(solution.input_data.num_vehicles))
        for j in range(solution.input_data.num_timesteps)
    ]
    y = [convert_to_interval([value / 1000 for value in cum_cp], is_y=True)]
    return PlotData(x=x, y=y)  # type: ignore # types are equivalent for lists of length 1


def get_energy_price_plot_data(solution: Solution) -> PlotData:
    assert solution.input_data.energy_price is not None
    x = convert_seconds_to_time(convert_to_interval(solution.input_data.time))
    y = [[ep * 3.6e6 for ep in convert_to_interval(solution.input_data.energy_price, is_y=True)]]
    return PlotData(x=x, y=y)


def get_energy_demand_plot_data(solution: Solution) -> PlotData:
    seconds = [solution.input_data.time[0] // 2]
    width = [solution.input_data.time[0]]
    for t1, t2 in zip(solution.input_data.time[:-1], solution.input_data.time[1:]):
        width.append(t2 - t1)
        seconds.append((t1 + t2) // 2)
    x = convert_seconds_to_time(seconds)
    y = [
        [ed / (1000 * dt) for ed, dt in zip(solution.input_data.energy_demand[i], width)]
        for i in range(solution.input_data.num_vehicles)
    ]
    width = [w * 1000 for w in width]
    return PlotData(x=x, y=y, width=width)


def get_depot_charge_bands(solution: Solution, index: int, color: str, opacity: float) -> list[dict]:
    assert index < solution.input_data.num_vehicles

    plot_data = get_state_of_energy_plot_data(solution)
    bands = []
    for x0, x1, dc in zip(plot_data.x[:-1], plot_data.x[1:], solution.input_data.depot_charge[index]):
        if dc:
            bands.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=opacity,
                    line_width=0,
                    layer="below",
                )
            )
    return bands


def color_wheel(index: int) -> str:
    trace_colors = [
        "#58A7C5",  # Darker Soft Sky Blue
        "#63B88D",  # Darker Light Mint
        "#E58D87",  # Darker Pale Coral
        "#A987D6",  # Darker Lavender Mist
        "#C7A94F",  # Darker Misty Gold
        "#BA8B86",  # Darker Dusty Rose
        "#4F9792",  # Darker Muted Teal
        "#9889D4",  # Darker Cool Lilac
        "#E4A87E",  # Darker Soft Apricot
        "#89AEB0",  # Darker Silver Blue
        "#90B9A9",  # Darker Powder Green
        "#CDBE97",  # Darker Creamy Sand
    ]
    return trace_colors[index % len(trace_colors)]


def convert_seconds_to_time(seconds: list[int] | list[float]) -> list[str]:
    start = datetime(2025, 1, 1)  # this date is arbitrary
    return [(start + timedelta(seconds=s)).isoformat() for s in seconds]


def convert_to_interval(values: list[int] | list[float], is_y: bool = False) -> list[int] | list[float]:
    intervals = [value for value in values for _ in range(2)]
    if is_y:
        return intervals
    if isinstance(values[0], int):
        return [0] + intervals[:-1]
    else:
        assert isinstance(values[0], float)
        return [0.0] + intervals[:-1]


def update_layout(fig: go.Figure, yaxis_title: str = "") -> None:
    fig.update_layout(
        xaxis=dict(
            type="date",
            tickformatstops=[
                dict(dtickrange=[None, 1000], value="%H:%M:%S.%L"),  # < 1s
                dict(dtickrange=[1000, 60000], value="%H:%M:%S"),  # < 1min
                dict(dtickrange=[60000, 3600000], value="%H:%M"),  # < 1h
                dict(dtickrange=[3600000, 86400000], value="%H:%M"),  # < 1 day
                dict(dtickrange=[86400000, 604800000], value="%a %H:%M"),  # < 1 week
                dict(dtickrange=[604800000, None], value="%b %d"),  # > 1 week
            ],
        ),
        xaxis_title="Time of Day",
        yaxis_title=yaxis_title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
