# import streamlit as st
# import pandas as pd
# import math
# from pathlib import Path

# # Set the title and favicon that appear in the Browser's tab bar.
# st.set_page_config(
#     page_title='GDP dashboard',
#     page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
# )

# # -----------------------------------------------------------------------------
# # Declare some useful functions.

# @st.cache_data
# def get_gdp_data():
#     """Grab GDP data from a CSV file.

#     This uses caching to avoid having to read the file every time. If we were
#     reading from an HTTP endpoint instead of a file, it's a good idea to set
#     a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
#     """

#     # Instead of a CSV on disk, you could read from an HTTP endpoint here too.
#     DATA_FILENAME = Path(__file__).parent/'data/gdp_data.csv'
#     raw_gdp_df = pd.read_csv(DATA_FILENAME)

#     MIN_YEAR = 1960
#     MAX_YEAR = 2022

#     # The data above has columns like:
#     # - Country Name
#     # - Country Code
#     # - [Stuff I don't care about]
#     # - GDP for 1960
#     # - GDP for 1961
#     # - GDP for 1962
#     # - ...
#     # - GDP for 2022
#     #
#     # ...but I want this instead:
#     # - Country Name
#     # - Country Code
#     # - Year
#     # - GDP
#     #
#     # So let's pivot all those year-columns into two: Year and GDP
#     gdp_df = raw_gdp_df.melt(
#         ['Country Code'],
#         [str(x) for x in range(MIN_YEAR, MAX_YEAR + 1)],
#         'Year',
#         'GDP',
#     )

#     # Convert years from string to integers
#     gdp_df['Year'] = pd.to_numeric(gdp_df['Year'])

#     return gdp_df

# gdp_df = get_gdp_data()

# # -----------------------------------------------------------------------------
# # Draw the actual page

# # Set the title that appears at the top of the page.
# '''
# # :earth_americas: GDP dashboard

# Browse GDP data from the [World Bank Open Data](https://data.worldbank.org/) website. As you'll
# notice, the data only goes to 2022 right now, and datapoints for certain years are often missing.
# But it's otherwise a great (and did I mention _free_?) source of data.
# '''

# # Add some spacing
# ''
# ''

# min_value = gdp_df['Year'].min()
# max_value = gdp_df['Year'].max()

# from_year, to_year = st.slider(
#     'Which years are you interested in?',
#     min_value=min_value,
#     max_value=max_value,
#     value=[min_value, max_value])

# countries = gdp_df['Country Code'].unique()

# if not len(countries):
#     st.warning("Select at least one country")

# selected_countries = st.multiselect(
#     'Which countries would you like to view?',
#     countries,
#     ['DEU', 'FRA', 'GBR', 'BRA', 'MEX', 'JPN'])

# ''
# ''
# ''

# # Filter the data
# filtered_gdp_df = gdp_df[
#     (gdp_df['Country Code'].isin(selected_countries))
#     & (gdp_df['Year'] <= to_year)
#     & (from_year <= gdp_df['Year'])
# ]

# st.header('GDP over time', divider='gray')

# ''

# st.line_chart(
#     filtered_gdp_df,
#     x='Year',
#     y='GDP',
#     color='Country Code',
# )

# ''
# ''


# first_year = gdp_df[gdp_df['Year'] == from_year]
# last_year = gdp_df[gdp_df['Year'] == to_year]

# st.header(f'GDP in {to_year}', divider='gray')

# ''

# cols = st.columns(4)

# for i, country in enumerate(selected_countries):
#     col = cols[i % len(cols)]

#     with col:
#         first_gdp = first_year[first_year['Country Code'] == country]['GDP'].iat[0] / 1000000000
#         last_gdp = last_year[last_year['Country Code'] == country]['GDP'].iat[0] / 1000000000

#         if math.isnan(first_gdp):
#             growth = 'n/a'
#             delta_color = 'off'
#         else:
#             growth = f'{last_gdp / first_gdp:,.2f}x'
#             delta_color = 'normal'

#         st.metric(
#             label=f'{country} GDP',
#             value=f'{last_gdp:,.0f}B',
#             delta=growth,
#             delta_color=delta_color
#         )


# import streamlit as st
# import pandas as pd

# from models.savings_model import predict_savings
# from models.system_size_model import predict_system_size
# from models.carbon_model import predict_carbon_reduction
# from models.lcoe_model import predict_lcoe
# from utils.preprocessing import clean_load_profile

# st.set_page_config(
#     page_title="Energy Forecasting Suite",
#     layout="wide"
# )

# st.title("Energy System Forecasting Suite")

# st.markdown("""
# This tool provides machine-learning based forecasts for:
# - 25-year savings  
# - Optimal solar system size  
# - Carbon emission reductions  
# - LCOE estimates  
# """)

# # ---------------- Sidebar Input Section ----------------

# with st.sidebar:
#     st.header("Input Parameters")

#     load_profile_file = st.file_uploader("Upload Load Profile (CSV)", type=["csv"])
#     tariff = st.number_input("Tariff Rate (₦/kWh)", min_value=0.0, value=120.0)
#     capex = st.number_input("Initial Cost (₦)", min_value=0.0, value=10_000_000.0)
#     opex = st.number_input("Annual Maintenance (₦)", min_value=0.0, value=200_000.0)
#     discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, max_value=30.0, value=8.0)
#     irradiance = st.number_input("Solar Irradiance (kWh/m²/day)", min_value=1.0, value=5.2)
#     carbon_factor = st.number_input("Grid CO₂ Factor (kg/kWh)", min_value=0.1, value=0.55)

#     run_button = st.button("Run Forecast", type="primary")

# # ---------------- Run Models ----------------

# if run_button:
#     if not load_profile_file:
#         st.error("Please upload a load profile first.")
#         st.stop()

#     df = pd.read_csv(load_profile_file)
#     df = clean_load_profile(df)

#     with st.spinner("Running ML predictions..."):
#         savings = predict_savings(df, tariff, capex, opex, discount_rate)
#         system_size = predict_system_size(df)
#         carbon = predict_carbon_reduction(df, carbon_factor)
#         lcoe_value = predict_lcoe(capex, opex, irradiance)

#     st.success("Forecast completed successfully!")

#     # ---------------- Results ----------------

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Optimal System Size")
#         st.metric("Recommended Solar PV Size", f"{system_size['pv_kw']} kW")
#         st.metric("Recommended Battery Storage", f"{system_size['battery_kwh']} kWh")

#     with col2:
#         st.subheader("Carbon Reduction")
#         st.metric("Annual CO₂ Avoided", f"{carbon['annual_tons']} tons")
#         st.metric("Lifetime CO₂ Avoided", f"{carbon['lifetime_tons']} tons")

#     st.subheader("25-Year Savings Forecast")
#     st.line_chart(savings["annual_savings"])

#     st.metric("Payback Period", f"{savings['payback_years']} years")
#     st.metric("Total 25-Year Savings", f"₦{savings['total_savings']:,.2f}")

#     st.subheader("Levelized Cost of Energy (LCOE)")
#     st.metric("LCOE", f"₦{lcoe_value:.2f} / kWh")



import streamlit as st
import pandas as pd

from models.savings_model import predict_savings
from models.system_size_model import predict_system_size
from models.carbon_model import predict_carbon_reduction
from models.lcoe_model import predict_lcoe
from utils.preprocessing import clean_load_profile

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="Energy Forecasting Suite",
    layout="wide",
)

# --------------------- HEADER --------------------------
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0;">⚡ Energy System Forecasting Suite</h1>
    <p style="text-align:center; font-size:18px; margin-top:4px;">
        Machine-learning driven forecasts for system sizing, carbon reduction, LCOE and long-term savings.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------- SIDEBAR --------------------------
with st.sidebar:
    st.header("⚙️ Input Parameters")

    st.markdown("### Load Profile")
    load_profile_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Hourly or sub-hourly load consumption data"
    )

    st.markdown("### Financial Inputs")
    tariff = st.number_input("Tariff Rate (₦/kWh)", min_value=0.0, value=120.0)
    capex = st.number_input("Initial Cost (₦)", min_value=0.0, value=10_000_000.0)
    opex = st.number_input("Annual Maintenance (₦)", min_value=0.0, value=200_000.0)
    discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, max_value=30.0, value=8.0)

    st.markdown("### Solar Parameters")
    irradiance = st.number_input("Solar Irradiance (kWh/m²/day)", min_value=1.0, value=5.2)
    carbon_factor = st.number_input("Grid CO₂ Factor (kg/kWh)", min_value=0.1, value=0.55)

    st.markdown("---")
    run_button = st.button("🚀 Run Forecast", type="primary")

# --------------------- LOGIC --------------------------
if run_button:
    if not load_profile_file:
        st.error("Please upload a load profile CSV before proceeding.")
        st.stop()

    df = pd.read_csv(load_profile_file)
    df = clean_load_profile(df)

    with st.spinner("Running intelligent energy predictions..."):
        savings = predict_savings(df, tariff, capex, opex, discount_rate)
        system_size = predict_system_size(df)
        carbon = predict_carbon_reduction(df, carbon_factor)
        lcoe_value = predict_lcoe(capex, opex, irradiance)

    st.success("Forecast completed successfully!")

    # --------------------- RESULTS -----------------------
    st.markdown("## 📊 Forecast Summary")

    # --- System Size + Carbon
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🔆 Optimal System Size")
        st.metric("Recommended Solar PV Size", f"{system_size['pv_kw']} kW")
        st.metric("Recommended Battery Storage", f"{system_size['battery_kwh']} kWh")

    with c2:
        st.markdown("### 🌍 Carbon Emission Reductions")
        st.metric("Annual CO₂ Avoided", f"{carbon['annual_tons']} tons")
        st.metric("Lifetime CO₂ Avoided", f"{carbon['lifetime_tons']} tons")

    # --- Savings Section
    st.markdown("## 💰 25-Year Financial Savings Forecast")

    st.line_chart(savings["annual_savings"])

    s1, s2 = st.columns(2)
    with s1:
        st.metric("Payback Period", f"{savings['payback_years']} years")
    with s2:
        st.metric("Total 25-Year Savings", f"₦{savings['total_savings']:,.2f}")

    # --- LCOE
    st.markdown("## ⚡ Levelized Cost of Energy (LCOE)")
    st.metric("LCOE", f"₦{lcoe_value:.2f} / kWh")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:grey;'>Model outputs generated using ML pipelines.</p>",
        unsafe_allow_html=True
    )
