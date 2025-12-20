# import numpy as np
# import pandas as pd
# from pathlib import Path
# import joblib

# _SAVINGS_PIPELINE = None
# _SAVINGS_FEATURES = None

# def _find_target_column(cols, keywords):
#     for kw in keywords:
#         for c in cols:
#             if kw.lower() in str(c).lower():
#                 return c
#     return None

# def _load_pipeline_and_features():
#     global _SAVINGS_PIPELINE, _SAVINGS_FEATURES
#     if _SAVINGS_PIPELINE is not None:
#         return _SAVINGS_PIPELINE, _SAVINGS_FEATURES

#     base = Path(__file__).resolve().parents[1]
#     model_path = base / "pv_model_outputs" / "rf_25yr_savings.pkl"
#     cleaned_csv = base / "pv_model_outputs" / "cleaned_pv_dataset.csv"

#     if model_path.exists():
#         try:
#             _SAVINGS_PIPELINE = joblib.load(model_path)
#         except Exception:
#             _SAVINGS_PIPELINE = None

#     if cleaned_csv.exists():
#         try:
#             df_clean = pd.read_csv(cleaned_csv)
#             numeric_cols = df_clean.select_dtypes(include=[float, int]).columns.tolist()
#             target_map = {
#                 "25yr_savings": ["savings at 25yrs", "savings at 25yr", "25yr savings", "savings at 25yrs lifespan"],
#                 "capacity": ["size/capacity", "existing pv system", "capacity", "kva", "kw", "system capacity"],
#                 "co2": ["emission", "co2", "carbon", "reduced carbon", "kgco2"],
#                 "lcoe": ["lcoe", "levelized cost", "levelised cost", "levelized cost of energy"]
#             }
#             detected = {k: _find_target_column(df_clean.columns, kws) for k, kws in target_map.items()}
#             exclude = [v for v in detected.values() if v is not None]
#             feature_cols = [c for c in numeric_cols if c not in exclude]
#             _SAVINGS_FEATURES = feature_cols
#         except Exception:
#             _SAVINGS_FEATURES = None
#     else:
#         _SAVINGS_FEATURES = None

#     return _SAVINGS_PIPELINE, _SAVINGS_FEATURES

# def predict_savings(df, tariff, capex, opex, discount):
#     """Predict 25-year savings using trained pipeline if available, otherwise fallback to placeholder."""
#     pipeline, feature_cols = _load_pipeline_and_features()
#     annual_load = df["load_kwh"].sum()

#     if pipeline is not None and feature_cols:
#         row = {}
#         for col in feature_cols:
#             lname = col.lower()
#             if any(k in lname for k in ("tariff", "price", "tarif")):
#                 row[col] = float(tariff)
#             elif any(k in lname for k in ("capex", "initial cost", "cost of investment", "initial")):
#                 row[col] = float(capex)
#             elif any(k in lname for k in ("opex", "maintenance", "annual opex", "opex")):
#                 row[col] = float(opex)
#             elif any(k in lname for k in ("discount", "rate")):
#                 row[col] = float(discount)
#             elif "annual" in lname or "year" in lname or "consumption" in lname or "energy" in lname or "kwh" in lname:
#                 row[col] = float(annual_load)
#             else:
#                 row[col] = np.nan
#         X = pd.DataFrame([row], columns=feature_cols)
#         try:
#             pred = pipeline.predict(X)
#             # Model predicts total 25yr savings. Distribute linearly for annual_savings for now.
#             total_savings = float(pred[0])
#             annual_savings = total_savings / 25.0
#             cumulative = [annual_savings * (i + 1) for i in range(25)]
#             payback_years = capex / annual_savings if annual_savings > 0 else float('inf')
#             return {
#                 "annual_savings": cumulative,
#                 "total_savings": total_savings,
#                 "payback_years": payback_years,
#                 "capex": capex,        # new
#                 "opex": opex           # new
#             }
#         except Exception:
#             pass

#     # Fallback placeholder
#     annual_savings = annual_load * tariff * 0.65
#     cumulative = [annual_savings * (i + 1) for i in range(25)]
#     return {
#         "annual_savings": cumulative,
#         "total_savings": cumulative[-1] - capex - (opex * 25),
#         "payback_years": capex / annual_savings if annual_savings > 0 else float('inf'),
#         "capex": capex,        # new
#         "opex": opex           # new
#     }


import numpy as np


def predict_savings(
    annual_load_kwh: float,
    tariff: float,
    capex: float,
    opex_annual: float,
    discount_rate: float,
    system_lifetime: int = 25,
    pv_degradation: float = 0.007
):
    """
    Bankable deterministic savings model.

    Parameters
    ----------
    annual_load_kwh : float
        Annual electricity consumption (kWh/year)
    tariff : float
        Grid tariff (₦/kWh)
    capex : float
        Initial investment cost (₦)
    opex_annual : float
        Annual O&M cost (₦/year)
    discount_rate : float
        Discount rate (%) e.g. 8.0
    system_lifetime : int
        Analysis period in years
    pv_degradation : float
        Annual PV degradation rate

    Returns
    -------
    dict
        annual_savings, total_savings, payback_years, npv
    """

    r = discount_rate / 100.0

    annual_energy_offset = annual_load_kwh
    base_annual_savings = annual_energy_offset * tariff

    cashflows = []
    discounted_cashflows = []
    cumulative_cashflow = -capex
    payback_year = None

    for year in range(1, system_lifetime + 1):
        degradation_factor = (1 - pv_degradation) ** (year - 1)
        gross_savings = base_annual_savings * degradation_factor
        net_savings = gross_savings - opex_annual

        discounted = net_savings / ((1 + r) ** year)

        cashflows.append(net_savings)
        discounted_cashflows.append(discounted)

        cumulative_cashflow += net_savings
        if cumulative_cashflow > 0 and payback_year is None:
            payback_year = year

    total_savings = sum(cashflows)
    npv = sum(discounted_cashflows) - capex

    return {
        "annual_savings": np.cumsum(cashflows).tolist(),
        "total_savings": total_savings,
        "payback_years": payback_year if payback_year else float("inf"),
        "npv": npv,
        "capex": capex,
        "opex": opex_annual
    }
