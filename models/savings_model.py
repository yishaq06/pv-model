import joblib
from pathlib import Path
from feature_builder import build_features

_CAPACITY_PIPELINE = None

def _load_pipeline():
    global _CAPACITY_PIPELINE
    if _CAPACITY_PIPELINE is not None:
        return _CAPACITY_PIPELINE
    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_capacity.pkl"
    if model_path.exists():
        _CAPACITY_PIPELINE = joblib.load(model_path)
    return _CAPACITY_PIPELINE

def predict_system_size(df_input):
    pipeline = _load_pipeline()
    try:
        df_features = build_features(df_input, 0, 0, 0, 0)
        pred = pipeline.predict(df_features)
        pv_kw = float(pred[0])
        battery_kwh = round(pv_kw * 2.5, 2)  # simple heuristic
        return {"pv_kw": round(pv_kw, 2), "battery_kwh": battery_kwh}
    except Exception:
        # fallback
        peak_load = df_input["load_kwh"].max()
        return {"pv_kw": round(peak_load * 1.3, 2), "battery_kwh": round(peak_load * 2.5, 2)}
