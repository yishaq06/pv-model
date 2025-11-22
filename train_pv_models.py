# train_pv_models.py
# Usage: python train_pv_models.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from pathlib import Path

INPUT_PATH = Path(__file__).parent/'data/MODEL_DEV_DATA_SHEET.xlsx'
OUTPUT_DIR = "./pv_model_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load and preprocess sheet (first row used as header) ---
df_raw = pd.read_excel(INPUT_PATH, header=0)
if df_raw.shape[0] < 2:
    raise RuntimeError("No data rows after header row. Inspect the Excel file.")

new_header = df_raw.iloc[0].fillna('').astype(str).tolist()
df = df_raw.iloc[1:].copy().reset_index(drop=True)
df.columns = new_header

# Convert numeric-like to numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Helper to find columns by keywords
def find_col(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None

target_map = {
    "25yr_savings": ["savings at 25yrs", "savings at 25yr", "25yr savings", "savings at 25yrs lifespan"],
    "capacity": ["size/capacity", "existing pv system", "capacity", "kva", "kw", "system capacity"],
    "co2": ["emission", "co2", "carbon", "reduced carbon", "kgco2"],
    "lcoe": ["lcoe", "levelized cost", "levelised cost", "levelized cost of energy"]
}

detected = {k: find_col(df.columns, kws) for k,kws in target_map.items()}
print("Detected targets:", detected)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = [v for v in detected.values() if v is not None]
feature_cols = [c for c in numeric_cols if c not in exclude]

print("Numeric columns found:", numeric_cols)
print("Feature columns chosen:", feature_cols)

# Save cleaned CSV for inspection
clean_csv = os.path.join(OUTPUT_DIR, "cleaned_pv_dataset.csv")
df.to_csv(clean_csv, index=False)
print("Saved cleaned dataset to:", clean_csv)

# Train helper
def train_for_target(target_col, model_name):
    if target_col is None:
        print(f"Skipping {model_name}: target column not found.")
        return None, None
    y = df[target_col]
    X = df[feature_cols].copy()
    mask = ~y.isna()
    X2 = X.loc[mask].reset_index(drop=True)
    y2 = y.loc[mask].reset_index(drop=True)
    nrows = X2.shape[0]
    if nrows < 3:
        print(f"Skipping {model_name}: insufficient rows ({nrows}).")
        return None, None
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    n_splits = min(3, max(2, nrows))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    try:
        scores = cross_val_score(pipeline, X2, y2, cv=cv, scoring=mae_scorer)
        cv_mae = -scores.mean()
    except Exception as e:
        print("CV failed:", e)
        cv_mae = None
    pipeline.fit(X2, y2)
    model_path = os.path.join(OUTPUT_DIR, f"rf_{model_name}.pkl")
    joblib.dump(pipeline, model_path)
    rf = pipeline.named_steps['rf']
    importances = dict(zip(feature_cols, rf.feature_importances_)) if hasattr(rf, "feature_importances_") else {}
    return pipeline, {"cv_mae": cv_mae, "n_rows": nrows, "model_path": model_path, "importances": importances}

# Run training for each target
results = {}
for key, col in detected.items():
    model, stats = train_for_target(col, key)
    results[key] = {"column": col, "model": model, "stats": stats}

# Summarize
summary = []
for k,v in results.items():
    stats = v["stats"]
    summary.append({
        "target": k,
        "column": v["column"],
        "trained": stats is not None,
        "rows_used": stats["n_rows"] if stats else None,
        "cv_mae": stats["cv_mae"] if stats else None,
        "model_path": stats["model_path"] if stats else None
    })
summary_df = pd.DataFrame(summary)
print("\nTraining summary:")
print(summary_df.to_string(index=False))

# Save summary csv
summary_df.to_csv(os.path.join(OUTPUT_DIR, "models_summary.csv"), index=False)
print("Saved models_summary.csv in:", OUTPUT_DIR)

# If models exist, list them
for f in os.listdir(OUTPUT_DIR):
    print(" -", f)

print("\nDone. Inspect cleaned CSV and models in the output folder.")
