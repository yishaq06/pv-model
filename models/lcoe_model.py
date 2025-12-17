from feature_builder import build_features

_LCOE_PIPELINE = None

import joblib
from pathlib import Path

def _load_pipeline():
    global _LCOE_PIPELINE
    if _LCOE_PIPELINE is not None:
        return _LCOE_PIPELINE
    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_lcoe.pkl"
    if model_path.exists():
        _LCOE_PIPELINE = joblib.load(model_path)
    return _LCOE_PIPELINE

def predict_lcoe(capex, opex, irradiance):
    pipeline = _load_pipeline()
    try:
        # Use a safe numeric column to build features
        df_input = pd.DataFrame({"dummy_load": [1]})
        df_features = build_features(df_input, 0, capex, opex, 0, irradiance)
        pred = pipeline.predict(df_features)
        return float(pred[0])
    except Exception:
        # fallback
        annual_energy = irradiance * 365 * 0.85
        lcoe = (capex + (opex * 25)) / (annual_energy * 25)
        return lcoe
