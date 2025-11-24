def compute_performance_ratio(pv_size_kw, irradiance, daily_load):
    """Compute PV system performance ratio (%) using expected vs actual energy yield."""

    # Expected annual generation
    expected_kwh = pv_size_kw * irradiance * 365 * 0.85  # 85% overall system efficiency

    # Actual energy used by load
    actual_kwh = daily_load * 365

    if expected_kwh == 0:
        return 0

    pr = (actual_kwh / expected_kwh) * 100
    return round(pr, 2)
