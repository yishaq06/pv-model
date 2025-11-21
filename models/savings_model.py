def predict_savings(df, tariff, capex, opex, discount):
    annual_load = df["load_kwh"].sum()
    annual_savings = annual_load * tariff * 0.65  # placeholder model

    cumulative = [annual_savings * (i + 1) for i in range(25)]

    return {
        "annual_savings": cumulative,
        "total_savings": cumulative[-1] - capex - (opex * 25),
        "payback_years": capex / annual_savings
    }
