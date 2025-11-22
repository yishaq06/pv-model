import numpy as np
from pathlib import Path
import joblib
import pandas as pd


_LCOE_PIPELINE = None
_LCOE_FEATURES = None


def _find_target_column(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def _load_pipeline_and_features():
    global _LCOE_PIPELINE, _LCOE_FEATURES
    if _LCOE_PIPELINE is not None:
        return _LCOE_PIPELINE, _LCOE_FEATURES

    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_lcoe.pkl"
    cleaned_csv = base / "pv_model_outputs" / "cleaned_pv_dataset.csv"

    if model_path.exists():
        try:
            _LCOE_PIPELINE = joblib.load(model_path)
        except Exception:
            _LCOE_PIPELINE = None

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
            _LCOE_FEATURES = feature_cols
        except Exception:
            _LCOE_FEATURES = None
    else:
        _LCOE_FEATURES = None

    return _LCOE_PIPELINE, _LCOE_FEATURES


def predict_lcoe(capex, opex, irradiance):
    """Predict LCOE using trained pipeline if available, otherwise fall back
    to the simple placeholder formula.
    """
    pipeline, feature_cols = _load_pipeline_and_features()

    if pipeline is not None and feature_cols:
        # Map available scalar inputs to feature columns by keyword matching
        row = {}
        for col in feature_cols:
            lname = col.lower()
            if any(k in lname for k in ("capex", "initial cost", "cost of investment", "initial")):
                row[col] = float(capex)
            elif any(k in lname for k in ("opex", "maintenance", "annual opex", "opex")):
                row[col] = float(opex)
            elif any(k in lname for k in ("irradiance", "irradiation", "solar", "insolation")):
                row[col] = float(irradiance)
            elif "tariff" in lname or "price" in lname or "tarif" in lname:
                # unknown tariff; set NaN so imputer may handle it
                row[col] = np.nan
            else:
                # give NaN for missing mappings; pipeline imputer should handle
                row[col] = np.nan

        X = pd.DataFrame([row], columns=feature_cols)
        try:
            pred = pipeline.predict(X)
            return float(pred[0])
        except Exception:
            pass

    # Fallback placeholder
    annual_energy = irradiance * 365 * 0.85
    lcoe = (capex + (opex * 25)) / (annual_energy * 25)
    return lcoe

