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
        # Use build_features for ML prediction
        df_features = build_features(
            df_input=df_input,
            tariff=0,  # irrelevant for CO2 model
            capex=0,
            opex=0,
            discount_rate=0
        )
        pred = pipeline.predict(df_features)
        annual_tons = float(pred[0])
        return {"annual_tons": annual_tons, "lifetime_tons": annual_tons * 25}

    except Exception:
        # fallback: use first numeric column as annual load
        try:
            annual_load = df_input.select_dtypes(include='number').iloc[:, 0].sum()
        except Exception:
            annual_load = 100 * 365  # rough default if input invalid

        annual_tons = round(annual_load * carbon_factor * 0.7 / 1000, 2)
        lifetime_tons = annual_tons * 25
        return {"annual_tons": annual_tons, "lifetime_tons": lifetime_tons}
