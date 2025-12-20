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

from models.savings_model_v1 import predict_savings
from models.system_size_model_v1 import predict_system_size
from models.carbon_model_v1 import predict_carbon_reduction
from models.lcoe_model_v1 import predict_lcoe
from models.performance_model import compute_performance_ratio

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Energy Forecasting Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Energy System Forecasting Suite")
st.markdown("""
**ML-powered forecasting for energy planning**.  
Visualize economic, environmental, and system performance metrics in a modern dashboard.
""")

CAPEX_PER_KW = 400_000.0  # ₦/kW
OPEX_PERCENT = 0.01       # 1% of CAPEX per year

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    with st.expander("📥 Input Parameters", expanded=True):

        st.markdown("**Energy Demand**")
        daily_load = st.number_input(
            "Daily Energy Consumption (kWh/day)",
            min_value=1.0, value=120.0,
            help="Average daily energy consumption in kWh"
        )
        peak_demand = st.number_input(
            "Peak Demand (kW)",
            min_value=0.1, value=25.0,
            help="Maximum expected instantaneous load in kW"
        )

        st.markdown("**Solar Resource**")
        irradiance = st.number_input(
            "Solar Irradiance (kWh/m²/day)",
            min_value=1.0, value=5.2,
            help="Average daily solar irradiance on PV panels"
        )

        st.markdown("**Financial Parameters**")
        tariff = st.number_input(
            "Tariff Rate (₦/kWh)", min_value=0.0, value=120.0,
            help="Cost of grid electricity per kWh"
        )
        discount_rate = st.number_input(
            "Discount Rate (%)", min_value=0.0, max_value=30.0, value=8.0,
            help="Discount rate for financial calculations"
        )

        st.markdown("**Environmental Parameters**")
        carbon_factor = st.number_input(
            "Grid CO₂ Factor (kg/kWh)", min_value=0.1, value=0.55,
            help="Carbon intensity of grid electricity"
        )

        st.markdown("---")
        run_button = st.button("🚀 Run Forecast", type="primary")

# ---------------- Run Models ----------------
if run_button:
    with st.spinner("Running ML-powered forecasting..."):

        annual_load_kwh = daily_load * 365
        df = pd.DataFrame({"load_kwh": [annual_load_kwh]})

        # ---- System Sizing ----
        system_size = predict_system_size(
            daily_load_kwh=daily_load,
            peak_demand_kw=peak_demand,
            irradiance_kwh_m2_day=irradiance
        )
        max_reasonable_pv = annual_load_kwh / 1000 * 2.0
        system_size["pv_kw"] = min(system_size["pv_kw"], max_reasonable_pv)

        # ---- CAPEX & OPEX ----
        capex = system_size["pv_kw"] * CAPEX_PER_KW
        opex = capex * OPEX_PERCENT
        capex = max(capex, 0)
        opex = min(opex, capex * 0.05)

        # ---- Savings Prediction ----
        savings = predict_savings(
            annual_load_kwh=annual_load_kwh,
            tariff=tariff,
            capex=capex,
            opex_annual=opex,
            discount_rate=discount_rate
        )
        carbon = predict_carbon_reduction(df, carbon_factor)
        lcoe_value = predict_lcoe(
            capex=capex, opex_annual=opex, irradiance=irradiance,
            discount_rate=discount_rate, pv_kw=system_size["pv_kw"]
        )
        performance = compute_performance_ratio(
            pv_size_kw=system_size["pv_kw"],
            irradiance=irradiance,
            daily_load=daily_load
        )

    st.success("✅ Forecast completed successfully!")


    # ---------------- Generate Executive Summary ----------------
    def generate_report(system_size, capex, opex, savings, carbon, lcoe, performance, tariff):
        report = f"Based on the current inputs, the recommended PV system size is {system_size['pv_kw']:.1f} kW "
        report += f"with a battery storage of {system_size['battery_kwh']:.1f} kWh. "

        # Performance Ratio
        if performance < 70:
            report += f"The system's performance ratio is {performance:.1f}%, which is below optimal. "
            report += "Improvements in panel orientation, cleaning, or higher-efficiency panels are advised. "
        else:
            report += f"The system performs efficiently with a performance ratio of {performance:.1f}%. "

        # Economic insights
        payback = savings["payback_years"]
        report += f"The initial investment cost is ₦{capex:,.0f} with annual OPEX of ₦{opex:,.0f}. "
        if payback > 15:
            report += f"However, the payback period is {payback:.1f} years, which is relatively long; consider reducing CAPEX or exploring incentives. "
        else:
            report += f"The payback period of {payback:.1f} years indicates a reasonable return on investment. "

        if lcoe > tariff:
            report += f"The levelized cost of electricity (LCOE) is ₦{lcoe:,.2f}/kWh, which exceeds the current grid tariff of ₦{tariff:,.2f}/kWh. "
            report += "Electricity from this PV system may be more expensive than grid electricity. "
        else:
            report += f"The LCOE is ₦{lcoe:,.2f}/kWh, below the grid tariff, making this system economically favorable. "

        # Environmental impact
        report += f"Environmentally, the system avoids {carbon['annual_tons']:.2f} tons of CO₂ annually and "
        report += f"{carbon['lifetime_tons']:.2f} tons over its lifetime, contributing positively to sustainability. "

        # Savings vs CAPEX
        total_savings = savings["total_savings"]
        if total_savings < capex:
            report += f"However, the total projected savings over 25 years is ₦{total_savings:,.0f}, which is below the initial CAPEX. "
            report += "System optimization could improve economic outcomes. "
        else:
            report += f"The total projected savings over 25 years is ₦{total_savings:,.0f}, exceeding the initial investment, highlighting long-term financial benefits. "

        return report


    # ---------------- KPI Cards ----------------
    st.header("📊 Key Forecast Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("PV Size", f"{system_size['pv_kw']:.1f} kW")
    kpi2.metric("Battery Storage", f"{system_size['battery_kwh']:.1f} kWh")
    kpi3.metric("Performance Ratio", f"{performance:.1f} %")
    kpi4.metric("LCOE", f"₦{lcoe_value:,.2f}/kWh")

    st.markdown("---")
    # ---------------- Economic & Environmental Columns ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Economic Outcomes")
        st.metric("CAPEX", f"₦{capex:,.0f}", delta_color="inverse")
        st.metric("Annual OPEX", f"₦{opex:,.0f}", delta_color="inverse")
        st.metric("Total 25-Year Savings", f"₦{savings['total_savings']:,.0f}", delta_color="normal")
        st.metric("Payback Period", f"{savings['payback_years']:.1f} yrs")

    with col2:
        st.subheader("🌱 Environmental Impact")
        st.metric("Annual CO₂ Avoided", f"{carbon['annual_tons']:.2f} tons", delta_color="normal")
        st.metric("Lifetime CO₂ Avoided", f"{carbon['lifetime_tons']:.2f} tons", delta_color="normal")

    # ---------------- Display Executive Summary ----------------
    st.subheader("📝 Executive Summary")
    report_text = generate_report(
        system_size, capex, opex, savings, carbon, lcoe_value, performance, tariff
    )

    st.info(report_text)

    # st.subheader("📈 Annual Savings Trend")
    # st.line_chart(savings["annual_savings"])

    # st.info(f"*CAPEX is fixed at ₦{CAPEX_PER_KW:,.0f}/kW. OPEX is {OPEX_PERCENT*100:.1f}% of CAPEX per year.*")
