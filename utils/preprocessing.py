import pandas as pd
import numpy as np

def clean_load_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes the load-profile dataset to ensure
    it is suitable for the ML forecasting models.

    Expected input structure:
    - A timestamp column (any name containing 'date', 'time', 'timestamp')
    - A load column (any name containing 'load' or 'kwh')

    Returns:
        A cleaned dataframe with:
        - 'timestamp' (datetime)
        - 'load_kwh' (float)
    """

    # ----------------------------------------------------
    # 1. Standardize column names
    # ----------------------------------------------------
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    # Identify timestamp column
    time_cols = [c for c in cols if "time" in c or "date" in c or "stamp" in c]
    if not time_cols:
        raise ValueError("No timestamp column found in uploaded file.")
    timestamp_col = time_cols[0]

    # Identify load column
    load_cols = [c for c in cols if "load" in c or "kwh" in c or "energy" in c]
    if not load_cols:
        raise ValueError("No load/energy column found in uploaded file.")
    load_col = load_cols[0]

    # ----------------------------------------------------
    # 2. Convert timestamp
    # ----------------------------------------------------
    df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # ----------------------------------------------------
    # 3. Convert load to numeric
    # ----------------------------------------------------
    df["load_kwh"] = pd.to_numeric(df[load_col], errors="coerce")
    df = df.dropna(subset=["load_kwh"])

    # Remove negative or impossible values
    df = df[df["load_kwh"] >= 0]

    # ----------------------------------------------------
    # 4. Handle duplicates
    # ----------------------------------------------------
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # ----------------------------------------------------
    # 5. Fill missing timestamps (optional)
    #    We enforce hourly resolution if timestamps are regular
    # ----------------------------------------------------
    df = df.set_index("timestamp")

    try:
        freq = df.index.inferred_freq
        if freq is None:
            # try forcing hourly frequency
            df = df.resample("1H").mean()
    except:
        df = df.resample("1H").mean()

    # ----------------------------------------------------
    # 6. Forward-fill missing load entries
    # ----------------------------------------------------
    df["load_kwh"] = df["load_kwh"].interpolate(method="linear").fillna(method="bfill")

    return df.reset_index()
