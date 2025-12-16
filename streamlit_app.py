# import streamlit as st
# import pandas as pd

# from models.savings_model import predict_savings
# from models.system_size_model import predict_system_size
# from models.carbon_model import predict_carbon_reduction
# from models.lcoe_model import predict_lcoe
# from models.performance_model import compute_performance_ratio

# st.set_page_config(
#     page_title="Energy Forecasting Suite",
#     layout="wide"
# )

# st.title("Energy System Forecasting Suite")

# st.markdown("""
# Machine-learning powered forecasting for energy system planning.  
# This dashboard generates the economic and environmental metrics required by the client.
# """)

# # ---------------- Sidebar ----------------

# with st.sidebar:
#     st.header("Input Parameters")

#     daily_load = st.number_input("Daily Energy Consumption (kWh/day)", min_value=1.0, value=120.0)
#     peak_demand = st.number_input("Peak Demand (kW)", min_value=0.1, value=25.0)
#     irradiance = st.number_input("Solar Irradiance (kWh/m²/day)", min_value=1.0, value=5.2)

#     tariff = st.number_input("Tariff Rate (₦/kWh)", min_value=0.0, value=120.0)
#     # capex = st.number_input("Initial Investment Cost – CAPEX (₦)", min_value=0.0, value=10_000_000.0)
#     # opex = st.number_input("Annual OPEX / Maintenance (₦)", min_value=0.0, value=200_000.0)
#     discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, max_value=30.0, value=8.0)

#     carbon_factor = st.number_input("Grid CO₂ Factor (kg/kWh)", min_value=0.1, value=0.55)

#     run_button = st.button("Run Forecast", type="primary")

# # ---------------- Run Models ----------------

# if run_button:

#     with st.spinner("Running ML-powered forecasting..."):

#         # Create synthetic df for compatibility with old models
#         df = pd.DataFrame({"load_kwh": [daily_load]})

#         system_size = predict_system_size(df)   # Still uses ML, but fed with synthetic df
#         # Determine CAPEX based on predicted PV size
#         CAPEX_PER_KW = 400_000  # NGN, configurable
#         capex = system_size["pv_kw"] * CAPEX_PER_KW

#         # Determine OPEX as % of CAPEX
#         OPEX_PERCENT = 0.01  # 1% per year
#         opex = capex * OPEX_PERCENT

#         savings = predict_savings(df, tariff, capex, opex, discount_rate)
#         carbon = predict_carbon_reduction(df, carbon_factor)
#         lcoe_value = predict_lcoe(capex, opex, irradiance)

#         performance = compute_performance_ratio(
#             pv_size_kw=system_size["pv_kw"],
#             irradiance=irradiance,
#             daily_load=daily_load
#         )

#     st.success("Forecast completed successfully!")

#     # ---------------- Layout ----------------
#     st.header("Forecast Results")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("System Capacity")
#         st.metric("Recommended PV Size", f"{system_size['pv_kw']} kW")
#         st.metric("Recommended Battery Storage", f"{system_size['battery_kwh']} kWh")

#         st.subheader("LCOE")
#         st.metric("LCOE", f"₦{lcoe_value:.2f} / kWh")

#     with col2:
#         st.subheader("Carbon Emission Reduction")
#         st.metric("Annual CO₂ Avoided", f"{carbon['annual_tons']} tons")
#         st.metric("Lifetime CO₂ Avoided", f"{carbon['lifetime_tons']} tons")

#         st.subheader("PV System Performance")
#         st.metric("Performance Ratio", f"{performance:.1f} %")

#     st.header("Economic Outcomes")
#     st.metric("Initial Investment Cost (CAPEX)", f"₦{savings['capex']:,.2f}")
#     st.metric("Annual OPEX / Maintenance", f"₦{savings['opex']:,.2f}")
#     st.metric("Total 25-Year Savings", f"₦{savings['total_savings']:,.2f}")
#     st.metric("Payback Period", f"{savings['payback_years']} years")

#     st.subheader("Annual Savings Trend")
#     st.line_chart(savings["annual_savings"])


import streamlit as st
import pandas as pd

from models.savings_model import predict_savings
from models.system_size_model import predict_system_size
from models.carbon_model import predict_carbon_reduction
from models.lcoe_model import predict_lcoe
from models.performance_model import compute_performance_ratio

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Energy Forecasting Suite",
    layout="wide"
)

st.title("Energy System Forecasting Suite")
st.markdown("""
Machine-learning powered forecasting for energy system planning.  
This dashboard generates economic and environmental metrics required by the client.
""")

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("Input Parameters")

    daily_load = st.number_input(
        "Daily Energy Consumption (kWh/day)", min_value=1.0, value=120.0
    )
    peak_demand = st.number_input(
        "Peak Demand (kW)", min_value=0.1, value=25.0
    )
    irradiance = st.number_input(
        "Solar Irradiance (kWh/m²/day)", min_value=1.0, value=5.2
    )

    tariff = st.number_input(
        "Tariff Rate (₦/kWh)", min_value=0.0, value=120.0
    )
    discount_rate = st.number_input(
        "Discount Rate (%)", min_value=0.0, max_value=30.0, value=8.0
    )
    carbon_factor = st.number_input(
        "Grid CO₂ Factor (kg/kWh)", min_value=0.1, value=0.55
    )

    # Optional: Client can confirm CAPEX per kW
    CAPEX_PER_KW = st.number_input(
        "CAPEX per kW (₦/kW)", min_value=0.0, value=400_000.0
    )
    OPEX_PERCENT = st.number_input(
        "OPEX (% of CAPEX per year)", min_value=0.0, max_value=5.0, value=1.0
    ) / 100  # convert % to decimal


    daily_load = 850.0        # kWh/day
    peak_demand = 120.0       # kW
    irradiance = 5.0          # kWh/m²/day
    tariff = 65.0             # ₦/kWh
    discount_rate = 10.0      # %
    carbon_factor = 0.55      # kg/kWh
    CAPEX_PER_KW = 400_000.0 # ₦/kW
    OPEX_PERCENT = 0.01       # 1% of CAPEX/year


    run_button = st.button("Run Forecast", type="primary")

# ---------------- Run Models ----------------
if run_button:
    with st.spinner("Running ML-powered forecasting..."):

        # Create synthetic df for compatibility with ML models
        # df = pd.DataFrame({"load_kwh": [daily_load]})
        annual_load_kwh = daily_load * 365  # convert kWh/day → kWh/year
        df = pd.DataFrame({"load_kwh": [annual_load_kwh]})


        # ---- System Sizing ----
        system_size = predict_system_size(df)  # ML predicted PV and battery size

        # ---- CAPEX & OPEX Derivation ----
        capex = system_size["pv_kw"] * CAPEX_PER_KW
        opex = capex * OPEX_PERCENT

        # ---- Savings Prediction ----
        savings = predict_savings(df, tariff, capex, opex, discount_rate)
        carbon = predict_carbon_reduction(df, carbon_factor)
        lcoe_value = predict_lcoe(capex, opex, irradiance)

        performance = compute_performance_ratio(
            pv_size_kw=system_size["pv_kw"],
            irradiance=irradiance,
            daily_load=daily_load
        )

    st.success("Forecast completed successfully!")

    # ---------------- Layout ----------------
    st.header("Forecast Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("System Capacity")
        st.metric("Recommended PV Size", f"{system_size['pv_kw']} kW")
        st.metric("Recommended Battery Storage", f"{system_size['battery_kwh']} kWh")

        st.subheader("LCOE")
        st.metric("LCOE", f"₦{lcoe_value:,.2f} / kWh")

    with col2:
        st.subheader("Carbon Emission Reduction")
        st.metric("Annual CO₂ Avoided", f"{carbon['annual_tons']:.2f} tons")
        st.metric("Lifetime CO₂ Avoided", f"{carbon['lifetime_tons']:.2f} tons")

        st.subheader("PV System Performance")
        st.metric("Performance Ratio", f"{performance:.1f} %")

    with col1:
        st.header("Economic Outcomes")
        st.metric("Initial Investment Cost (CAPEX)", f"₦{capex:,.2f}")
        st.metric("Annual OPEX / Maintenance", f"₦{opex:,.2f}")

    with col2:
        st.header("")
        st.metric("Total 25-Year Savings", f"₦{savings['total_savings']:,.2f}")
        st.metric("Payback Period", f"{savings['payback_years']:.1f} years")

    st.subheader("Annual Savings Trend")
    st.line_chart(savings["annual_savings"])

    st.markdown(
        f"*Note: CAPEX is calculated deterministically as ₦{CAPEX_PER_KW:,.0f}/kW of predicted PV size. "
        f"OPEX is {OPEX_PERCENT*100:.1f}% of CAPEX per year.*"
    )
