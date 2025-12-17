import joblib
from pathlib import Path
from feature_builder import build_features

_SAVINGS_PIPELINE = None

def _load_pipeline():
    global _SAVINGS_PIPELINE
    if _SAVINGS_PIPELINE is not None:
        return _SAVINGS_PIPELINE
    base = Path(__file__).resolve().parents[1]
    model_path = base / "pv_model_outputs" / "rf_25yr_savings.pkl"
    if model_path.exists():
        _SAVINGS_PIPELINE = joblib.load(model_path)
    return _SAVINGS_PIPELINE

def predict_savings(df_input, tariff, capex, opex, discount_rate):
    pipeline = _load_pipeline()
    try:
        df_features = build_features(df_input, tariff, capex, opex, discount_rate)
        pred = pipeline.predict(df_features)
        total_savings = float(pred[0])
        annual_savings = total_savings / 25
        cumulative = [annual_savings * (i + 1) for i in range(25)]
        return {
            "annual_savings": cumulative,
            "total_savings": total_savings,
            "payback_years": capex / annual_savings if annual_savings > 0 else float("inf"),
            "capex": capex,
            "opex": opex
        }
    except Exception:
        # fallback
        annual_load = df_input["load_kwh"].sum()
        annual_savings = annual_load * tariff * 0.65
        cumulative = [annual_savings * (i + 1) for i in range(25)]
        return {
            "annual_savings": cumulative,
            "total_savings": cumulative[-1] - capex - opex * 25,
            "payback_years": capex / annual_savings if annual_savings > 0 else float('inf'),
            "capex": capex,
            "opex": opex
        }
