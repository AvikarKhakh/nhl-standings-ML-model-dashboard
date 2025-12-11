import math
import pickle
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = Path("data/master_team_with_points.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "points_rf.pkl"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = REPORTS_DIR / "metrics_points.json"

# Adjust these if you ever want to change train/test split
TRAIN_SEASONS = list(range(2013, 2024))  # 2013–2023
TEST_SEASON = 2024                       # Held-out season


def main():
    print(f"Loading merged dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Basic sanity check
    required_cols = {"season", "team", "points"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Make sure season is int
    df["season"] = df["season"].astype(int)

    # === Define feature columns ===
    # Use advanced stat columns ONLY.
    # Exclude:
    #   - season, team, nhl_tricode (identifiers)
    #   - points (target)
    #   - wins, losses, ot  <-- REMOVE LEAKAGE
    exclude_cols = {
        "season",
        "team",
        "points",
        "nhl_tricode",
        "wins",
        "losses",
        "ot",
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols
    ]

    print("Feature columns being used:")
    for col in feature_cols:
        print(f"  - {col}")

    # Train/test split based on season
    train_mask = df["season"].isin(TRAIN_SEASONS)
    test_mask = df["season"] == TEST_SEASON

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["points"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["points"].values

    print(f"\nTraining seasons: {sorted(train_df['season'].unique())}")
    print(f"Test season: {TEST_SEASON}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}\n")

    # === Train RandomForestRegressor (tuned hyperparameters) ===
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    print("=== Model Performance ===")
    print(f"Train R^2:  {train_r2:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    
    print(f"Test R^2:   {test_r2:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}")
    metrics = {
    "rmse": round(float(test_rmse), 3),
    "r2": round(float(test_r2), 3),
    "n_train": int(len(train_df)),
    "n_test": int(len(test_df)),
    "n_samples": int(len(train_df) + len(test_df)),
    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics JSON to: {METRICS_PATH}")

    # Save JUST the sklearn model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"\nSaved RandomForestRegressor model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()