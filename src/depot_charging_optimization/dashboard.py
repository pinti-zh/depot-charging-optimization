import io
import json

import streamlit as st

from depot_charging_optimization.data_models import Solution
from depot_charging_optimization.plots import state_of_energy_trajectories_figure, cumulative_charging_power_figure, detail_figure, energy_price_figure, input_data_figure, color_wheel


def get_figure_buffer(figure, title=""):
    buf = io.BytesIO()
    figure.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black'),
        title=title,
        title_font_color='black',
        legend_font_color='black',
        margin=dict(l=10, r=10, t=40, b=10),
    )
    figure.update_xaxes(showgrid=True, gridcolor='lightgrey')
    figure.update_yaxes(showgrid=True, gridcolor='lightgrey')
    figure.write_image(buf, format="png")
    buf.seek(0)
    return buf


def get_names(solution):
    names = []
    battery_index = 1
    bus_index = 1
    for dc in solution.input_data.depot_charge:
        if all(dc):
            names.append(f"Battery {battery_index}")
            battery_index += 1
        else:
            names.append(f"Bus {bus_index}")
            bus_index += 1
    return names


with open("outputs/solutions/solution.json", "r") as f:
    solution = Solution.model_validate(json.load(f))

names = get_names(solution)

# ==========================================
#  DASHBOARD
# ==========================================
st.title("⚡ Depot Charging Optimization ⚡")

st.subheader("Outputs")
output_tab_names = ["State of Energy", "Charging Power", "Detail"]
tab_state_of_energy, tab_charging_power, tab_detail = st.tabs(output_tab_names)

with tab_state_of_energy:
    st.header("State of Energy Trajectories")
    soe_selection = []
    with st.container(horizontal=True):
        for i in range(solution.input_data.num_vehicles):
            with st.container(horizontal=True, border=True, width=150):
                soe_selection.append(st.checkbox(" ", key=f"soe-{i}", value=True))
                st.html(f"<span style='color: {color_wheel(i)}; font-weight: bold;'>{names[i]}</span>")
    soe_figure = state_of_energy_trajectories_figure(solution, names, soe_selection)
    st.plotly_chart(soe_figure, width="stretch")

with tab_charging_power:
    st.header("Cumulative Charging Power")
    cp_selection = []
    with st.container(horizontal=True):
        for i in range(solution.input_data.num_vehicles):
            with st.container(horizontal=True, border=True, width=150):
                cp_selection.append(st.checkbox(" ", key=f"cp-{i}", value=True))
                st.html(f"<span style='color: {color_wheel(i)}; font-weight: bold;'>{names[i]}</span>")
    cp_figure = cumulative_charging_power_figure(solution, names, cp_selection)
    st.plotly_chart(cp_figure, width="stretch")

with tab_detail:
    st.header("Vehicle Detail")
    detail_index = st.selectbox("Vehicle", names)
    show_lower_envelope = st.checkbox("Show Lower Envelope", value=False)
    dt_figure = detail_figure(solution, names.index(detail_index), show_lower_envelope=show_lower_envelope)
    st.plotly_chart(dt_figure, width="stretch")

st.subheader("Inputs")
input_tabs_names = ["Energy Demand", "Energy Price"]
tab_energy_demand, tab_energy_price = st.tabs(input_tabs_names)

with tab_energy_demand:
    st.header("Energy Demand")
    input_index = st.selectbox("Input", names)
    ipt_figure = input_data_figure(solution, names.index(input_index))
    st.plotly_chart(ipt_figure, width="stretch")

with tab_energy_price:
    st.header("Energy Price")
    ep_figure = energy_price_figure(solution)
    st.plotly_chart(ep_figure, width="stretch")

with tab_state_of_energy:
    st.download_button(
        label="Download Plot",
        data=get_figure_buffer(soe_figure, title="State of Energy Trajectories"),
        file_name="state_of_energy_trajectories.png",
        mime="image/png"
    )

with tab_charging_power:
    st.download_button(
        label="Download Plot",
        data=get_figure_buffer(cp_figure, title="Cumulative Charging Power"),
        file_name="cumulative_charging_power.png",
        mime="image/png"
    )

with tab_detail:
    st.download_button(
        label="Download Plot",
        data=get_figure_buffer(dt_figure, title="Vehicle Detail"),
        file_name="vehicle_detail.png",
        mime="image/png"
    )

with tab_energy_demand:
    st.download_button(
        label="Download Plot",
        data=get_figure_buffer(ipt_figure, title="Energy Demand"),
        file_name="energy_demand.png",
        mime="image/png"
    )

with tab_energy_price:
    st.download_button(
        label="Download Plot",
        data=get_figure_buffer(ep_figure, title="Energy Price"),
        file_name="energy_price.png",
        mime="image/png"
    )
