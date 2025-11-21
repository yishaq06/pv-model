def predict_carbon_reduction(df, carbon_factor):
    annual_load = df["load_kwh"].sum()
    annual_carbon = annual_load * carbon_factor * 0.7  # placeholder

    return {
        "annual_tons": round(annual_carbon / 1000, 2),
        "lifetime_tons": round((annual_carbon * 25) / 1000, 2)
    }
