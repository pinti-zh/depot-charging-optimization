import streamlit as st
import json

from depot_charging_optimization.data_models import Solution
from depot_charging_optimization.plots import state_of_energy_trajectories_figure, cumulative_charging_power_figure, detail_figure, energy_price_figure, input_data_figure


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
    soe_selection = [st.checkbox(names[i], value=True, key=f"soe-{i}") for i in range(solution.input_data.num_vehicles)]
    st.plotly_chart(state_of_energy_trajectories_figure(solution, names, soe_selection), width="stretch")

with tab_charging_power:
    st.header("Cumulative Charging Power")
    cp_selection = [st.checkbox(names[i], value=True, key=f"cp-{i}") for i in range(solution.input_data.num_vehicles)]
    st.plotly_chart(cumulative_charging_power_figure(solution, names, cp_selection), width="stretch")

with tab_detail:
    st.header("Vehicle Detail")
    detail_index = st.selectbox("Vehicle", names)
    show_lower_envelope = st.checkbox("Show Lower Envelope", value=False)
    st.plotly_chart(detail_figure(solution, names.index(detail_index), show_lower_envelope=show_lower_envelope), width="stretch")

st.subheader("Inputs")
input_tabs_names = ["Energy Demand", "Energy Price"]
tab_energy_demand, tab_energy_price = st.tabs(input_tabs_names)

with tab_energy_demand:
    st.header("Energy Demand")
    input_index = st.selectbox("Input", names)
    st.plotly_chart(input_data_figure(solution, names.index(input_index)), width="stretch")

with tab_energy_price:
    st.header("Energy Price")
    st.plotly_chart(energy_price_figure(solution), width="stretch")
