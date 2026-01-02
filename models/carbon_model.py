import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


_CO2_PIPELINE = None
_CO2_FEATURES = None


def _find_target_column(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def _load_pipeline_and_features():
    global _CO2_PIPELINE, _CO2_FEATURES
    if _CO2_PIPELINE is not None:
        return _CO2_PIPELINE, _CO2_FEATURES

    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_co2.pkl"
    cleaned_csv = base / "pv_model_outputs" / "cleaned_pv_dataset.csv"

    if model_path.exists():
        try:
            _CO2_PIPELINE = joblib.load(model_path)
        except Exception:
            _CO2_PIPELINE = None

    # Try to reconstruct feature columns from cleaned csv (same logic as training)
    if cleaned_csv.exists():
        try:
            df_clean = pd.read_csv(cleaned_csv)
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
            # detect targets by keywords used in training
            target_map = {
                "25yr_savings": ["savings at 25yrs", "savings at 25yr", "25yr savings", "savings at 25yrs lifespan"],
                "capacity": ["size/capacity", "existing pv system", "capacity", "kva", "kw", "system capacity"],
                "co2": ["emission", "co2", "carbon", "reduced carbon", "kgco2"],
                "lcoe": ["lcoe", "levelized cost", "levelised cost", "levelized cost of energy"]
            }
            detected = {k: _find_target_column(df_clean.columns, kws) for k, kws in target_map.items()}
            exclude = [v for v in detected.values() if v is not None]
            feature_cols = [c for c in numeric_cols if c not in exclude]
            _CO2_FEATURES = feature_cols
        except Exception:
            _CO2_FEATURES = None
    else:
        _CO2_FEATURES = None

    return _CO2_PIPELINE, _CO2_FEATURES


def _aggregate_for_feature(name, df):
    """Heuristic aggregation for a timeseries `df` to produce a single value
    for a feature called `name`. Uses simple rules (sum/mean/max) based on
    keywords. Returns np.nan if no sensible aggregation is found.
    """
    lname = name.lower()
    if "annual" in lname or "year" in lname or "total" in lname or "consumption" in lname:
        return float(df["load_kwh"].sum())
    if "energy" in lname or "generation" in lname:
        return float(df["load_kwh"].sum())
    if "peak" in lname or "max" in lname:
        return float(df["load_kwh"].max())
    if "mean" in lname or "avg" in lname or "average" in lname:
        return float(df["load_kwh"].mean())
    # fallback: if the feature name contains 'kwh' or 'kw' use sum
    if "kwh" in lname or "kw" in lname:
        return float(df["load_kwh"].sum())
    return np.nan


def predict_carbon_reduction(df, carbon_factor):
    """Predict annual and lifetime CO2 reduction.

    This will attempt to load a trained pipeline at
    `pv_model_outputs/rf_co2.pkl` and reconstruct the feature vector from
    `pv_model_outputs/cleaned_pv_dataset.csv`. If that fails, it falls back
    to the original placeholder formula.
    """
    annual_load = df["load_kwh"].sum()

    pipeline, feature_cols = _load_pipeline_and_features()

    if pipeline is not None and feature_cols:
        # build a single-row DataFrame with columns in the same order
        row = {}
        for col in feature_cols:
            # try to find a matching column in the incoming df (case-insensitive)
            matches = [c for c in df.columns if col.lower() in c.lower()]
            if matches:
                # aggregate the matched column
                try:
                    series = pd.to_numeric(df[matches[0]], errors="coerce")
                    if series.isnull().all():
                        val = np.nan
                    else:
                        val = float(series.sum()) if "kwh" in matches[0].lower() or "energy" in matches[0].lower() else float(series.mean())
                except Exception:
                    val = np.nan
            else:
                # use heuristic aggregation from load profile
                val = _aggregate_for_feature(col, df)
            row[col] = val

        X = pd.DataFrame([row], columns=feature_cols)
        try:
            pred = pipeline.predict(X)
            # model was trained on 'Reduced carbon emission  (tonnes) ...'
            annual_tons = float(pred[0])
            lifetime_tons = round(annual_tons * 25, 2)
            return {"annual_tons": round(annual_tons, 2), "lifetime_tons": round(lifetime_tons, 2)}
        except Exception:
            # fallback to placeholder
            pass

    # Fallback placeholder (keeps previous behaviour). Assume carbon_factor in kg/kWh
    annual_carbon_kg = annual_load * carbon_factor * 0.7
    annual_tons = round(annual_carbon_kg / 1000, 2)
    lifetime_tons = round((annual_carbon_kg * 25) / 1000, 2)

    return {"annual_tons": annual_tons, "lifetime_tons": lifetime_tons}
