import numpy as np
import pandas as pd
from pathlib import Path
import joblib

_CAPACITY_PIPELINE = None
_CAPACITY_FEATURES = None


def _find_target_column(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def _load_pipeline_and_features():
    global _CAPACITY_PIPELINE, _CAPACITY_FEATURES
    if _CAPACITY_PIPELINE is not None:
        return _CAPACITY_PIPELINE, _CAPACITY_FEATURES

    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_capacity.pkl"
    cleaned_csv = base / "pv_model_outputs" / "cleaned_pv_dataset.csv"

    if model_path.exists():
        try:
            _CAPACITY_PIPELINE = joblib.load(model_path)
        except Exception:
            _CAPACITY_PIPELINE = None

    if cleaned_csv.exists():
        try:
            df_clean = pd.read_csv(cleaned_csv)
            numeric_cols = df_clean.select_dtypes(
                include=[float, int]).columns.tolist()
            target_map = {
                "25yr_savings": ["savings at 25yrs", "savings at 25yr", "25yr savings", "savings at 25yrs lifespan"],
                "capacity": ["size/capacity", "existing pv system", "capacity", "kva", "kw", "system capacity"],
                "co2": ["emission", "co2", "carbon", "reduced carbon", "kgco2"],
                "lcoe": ["lcoe", "levelized cost", "levelised cost", "levelized cost of energy"]
            }
            detected = {k: _find_target_column(
                df_clean.columns, kws) for k, kws in target_map.items()}
            exclude = [v for v in detected.values() if v is not None]
            feature_cols = [c for c in numeric_cols if c not in exclude]
            _CAPACITY_FEATURES = feature_cols
        except Exception:
            _CAPACITY_FEATURES = None
    else:
        _CAPACITY_FEATURES = None

    return _CAPACITY_PIPELINE, _CAPACITY_FEATURES

# def predict_system_size(df):
#     pipeline, feature_cols = _load_pipeline_and_features()
#     peak_load = df["load_kwh"].max()

#     if pipeline is not None and feature_cols:
#         row = {}
#         for col in feature_cols:
#             lname = col.lower()
#             if "peak" in lname or "max" in lname or "load" in lname or "kwh" in lname or "energy" in lname:
#                 row[col] = float(peak_load)
#             else:
#                 row[col] = np.nan
#         X = pd.DataFrame([row], columns=feature_cols)
#         try:
#             pred = pipeline.predict(X)
#             # Model predicts system size (kW). Use battery sizing as a function of PV size for now.
#             pv_kw = float(pred[0])
#             battery_kwh = round(pv_kw * 2.5, 2)
#             return {"pv_kw": round(pv_kw, 2), "battery_kwh": battery_kwh}
#         except Exception:
#             pass

#     # Fallback placeholder
#     return {
#         "pv_kw": round(peak_load * 1.3, 2),
#         "battery_kwh": round(peak_load * 2.5, 2)
#     }

# models/system_size_model_v1.py


def predict_system_size(
    daily_load_kwh: float,
    peak_demand_kw: float,
    irradiance_kwh_m2_day: float,
    system_autonomy_days: float = 1.0,
    night_fraction: float = 0.5,
    depth_of_discharge: float = 0.8,
    battery_efficiency: float = 0.9,
    pv_efficiency: float = 0.18,
    derating_factor: float = 0.8
):
    """
    Deterministic PV + battery sizing.

    Parameters
    ----------
    daily_load_kwh : float
        Daily energy consumption (kWh/day)
    peak_demand_kw : float
        Peak demand (kW)
    irradiance_kwh_m2_day : float
        Average daily solar irradiance
    system_autonomy_days : float
        Days of autonomy for battery
    night_fraction : float
        Fraction of daily load that occurs at night
    depth_of_discharge : float
        Usable battery fraction
    battery_efficiency : float
        Round-trip efficiency of battery
    pv_efficiency : float
        PV module efficiency
    derating_factor : float
        Overall system losses (inverter, wiring, temperature)

    Returns
    -------
    dict
        pv_kw : Recommended PV size (kW)
        battery_kwh : Recommended battery capacity (kWh)
    """

    # ---- PV Sizing ----
    # Daily PV generation required ≈ daily_load / derating
    required_daily_gen = daily_load_kwh / derating_factor

    # PV array area (m²) = required_daily_gen / irradiance / efficiency
    # PV size (kW) = irradiance × area × efficiency → simplified:
    pv_kw = required_daily_gen / irradiance_kwh_m2_day

    # Limit PV to peak demand if desired
    pv_kw = max(pv_kw, peak_demand_kw)
    pv_kw = round(pv_kw, 2)

    # ---- Battery Sizing ----
    # Night load fraction × autonomy days, corrected for DoD and efficiency
    usable_energy = daily_load_kwh * night_fraction * system_autonomy_days
    battery_kwh = round(
        usable_energy / (depth_of_discharge * battery_efficiency), 1)

    return {"pv_kw": pv_kw, "battery_kwh": battery_kwh}
