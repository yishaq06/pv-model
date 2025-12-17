import pandas as pd

def build_features(df_input, tariff, capex, opex, discount_rate, irradiance=None):
    """
    Construct a DataFrame with all features required by trained PV models:
    - 25-year savings
    - System size
    - CO2 reduction
    - LCOE
    
    Parameters
    ----------
    df_input : pd.DataFrame
        Should contain at least `load_kwh` per day
    tariff : float
        Electricity price (₦/kWh)
    capex : float
        Initial investment cost (₦)
    opex : float
        Annual OPEX (₦)
    discount_rate : float
        Discount rate (%) - optional
    irradiance : float, optional
        Daily solar irradiance (kWh/m²/day)
    
    Returns
    -------
    pd.DataFrame
        Single-row feature DataFrame for ML models
    """
    daily_load = df_input["load_kwh"].sum()
    peak_load = df_input["load_kwh"].max()
    irradiance = irradiance or 5.0  # fallback

    # PV generation estimate
    pv_generation_kwh = daily_load * 0.85
    grid_energy_cost = daily_load * tariff
    net_annual_pv_cost = capex / 25 + opex

    features = {
        # Common numeric features
        "initial cost of investment/ - C": capex,
        "Annual energy consumption from the grid (kWh)            D": daily_load,
        "annual Energy Generation from Pv system(KWh)            E": pv_generation_kwh,
        "Electricity tarrif (₦)/KWh              G": tariff,
        "Cost of energy consumed from the Grid kWh/year (₦)      = G*D": grid_energy_cost,
        "annual cost of energy produced by Pv system                J": capex / 25,
        "cost of maintenance= 1% of J                       K": opex,
        "net Annual costof energy produced by Pv system   =J-K": net_annual_pv_cost,
        "Payback  period -Yrs (M) = C/L": capex / (daily_load * tariff * 0.65),
        "NGAF Nigeria (N) KgCO2e/kWh": 0.5346,
        "Annual Reduced carbon KgCO2e/kWh  (P) = N*E": 0.5346 * pv_generation_kwh,
        "Reduced carbon emission  (tonnes) (Q) P/1000": 0.5346 * pv_generation_kwh / 1000,
        "total lifecycle Reduced carbon emission  (tonnes),LCC at 25yrs-(N)         (R)": 0.5346 * pv_generation_kwh / 1000 * 25,
        "total life time energy production (KWh)  E*25(S)": pv_generation_kwh * 25,
        "LCOE-N/unit (T)": (capex + opex * 25) / (pv_generation_kwh * 25),
        "Cost of energy consumed from the Grid kWh/30yrs (₦)      ( U)": grid_energy_cost * 30,
        "Savings at 25yrs lifespan -(₦) -(V)           (J-K)": (daily_load * tariff * 25 * 0.65) - capex - opex * 25,
        # Peak load for system sizing
        "peak_load": peak_load,
        # Irradiance for LCOE
        "irradiance": irradiance
    }

    return pd.DataFrame([features])
