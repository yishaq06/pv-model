def predict_lcoe(capex, opex, irradiance):
    annual_energy = irradiance * 365 * 0.85  # placeholder
    lcoe = (capex + (opex * 25)) / (annual_energy * 25)
    return lcoe
