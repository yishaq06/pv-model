# carbon_model.py
import joblib
from pathlib import Path
from feature_builder import build_features

_CO2_PIPELINE = None

def _load_pipeline():
    global _CO2_PIPELINE
    if _CO2_PIPELINE is not None:
        return _CO2_PIPELINE
    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_co2.pkl"
    if model_path.exists():
        _CO2_PIPELINE = joblib.load(model_path)
    return _CO2_PIPELINE

def predict_carbon_reduction(df_input, carbon_factor=0.5346):
    pipeline = _load_pipeline()
    try:
        df_features = build_features(df_input, 0, 0, 0, 0)
        pred = pipeline.predict(df_features)
        annual_tons = float(pred[0])
        return {"annual_tons": annual_tons, "lifetime_tons": annual_tons * 25}
    except Exception:
        # fallback
        annual_load = df_input["load_kwh"].sum()
        annual_tons = round(annual_load * carbon_factor * 0.7 / 1000, 2)
        lifetime_tons = annual_tons * 25
        return {"annual_tons": annual_tons, "lifetime_tons": lifetime_tons}
