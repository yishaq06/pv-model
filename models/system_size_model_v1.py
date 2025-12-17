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
            numeric_cols = df_clean.select_dtypes(include=[float, int]).columns.tolist()
            target_map = {
                "25yr_savings": ["savings at 25yrs", "savings at 25yr", "25yr savings", "savings at 25yrs lifespan"],
                "capacity": ["size/capacity", "existing pv system", "capacity", "kva", "kw", "system capacity"],
                "co2": ["emission", "co2", "carbon", "reduced carbon", "kgco2"],
                "lcoe": ["lcoe", "levelized cost", "levelised cost", "levelized cost of energy"]
            }
            detected = {k: _find_target_column(df_clean.columns, kws) for k, kws in target_map.items()}
            exclude = [v for v in detected.values() if v is not None]
            feature_cols = [c for c in numeric_cols if c not in exclude]
            _CAPACITY_FEATURES = feature_cols
        except Exception:
            _CAPACITY_FEATURES = None
    else:
        _CAPACITY_FEATURES = None

    return _CAPACITY_PIPELINE, _CAPACITY_FEATURES

def predict_system_size(df):
    pipeline, feature_cols = _load_pipeline_and_features()
    peak_load = df["load_kwh"].max()

    if pipeline is not None and feature_cols:
        row = {}
        for col in feature_cols:
            lname = col.lower()
            if "peak" in lname or "max" in lname or "load" in lname or "kwh" in lname or "energy" in lname:
                row[col] = float(peak_load)
            else:
                row[col] = np.nan
        X = pd.DataFrame([row], columns=feature_cols)
        try:
            pred = pipeline.predict(X)
            # Model predicts system size (kW). Use battery sizing as a function of PV size for now.
            pv_kw = float(pred[0])
            battery_kwh = round(pv_kw * 2.5, 2)
            return {"pv_kw": round(pv_kw, 2), "battery_kwh": battery_kwh}
        except Exception:
            pass

    # Fallback placeholder
    return {
        "pv_kw": round(peak_load * 1.3, 2),
        "battery_kwh": round(peak_load * 2.5, 2)
    }
