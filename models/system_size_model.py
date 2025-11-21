def predict_system_size(df):
    peak_load = df["load_kwh"].max()

    return {
        "pv_kw": round(peak_load * 1.3, 2),
        "battery_kwh": round(peak_load * 2.5, 2)
    }
