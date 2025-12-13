import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = "data/master_team_with_points.csv"
MODEL_PATH = "models/points_rf.pkl"

# Features we want to use
FEATURE_COLS = [
    "xgf60","xga60","ev_xgf60","ev_xga60","pp_xgf60","pp_xga60",
    "pk_xgf60","pk_xga60","hdf","hda","hdf_percent","xgoals_percentage",
    "corsi_percentage","fenwick_percentage","penalties_drawn_per60",
    "penalties_taken_per60","penalty_diff","gsa_per60","save_percent",
    "hd_save_percent"
]

TARGET = "points"

def main():
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Sort for time-aware splits
    df = df.sort_values(["season", "team"])

    # Train seasons: 2013–2023
    train_df = df[df["season"] < 2024]
    test_df = df[df["season"] == 2024]

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET]

    print("Training size:", len(X_train))
    print("Testing size:", len(X_test))

    tscv = TimeSeriesSplit(n_splits=5)

    # Hyperparameter grid
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestRegressor(random_state=42)

    print("Starting GridSearchCV (may take ~1 minute)...")

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\n=== Best Hyperparameters ===")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    # Fix for older sklearn versions that don't have squared=False
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)


    print("\n=== Test Performance After Tuning ===")
    print(f"Test R^2: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")

    # Save tuned model bundle
    bundle = {
        "model": best_model,
        "features": FEATURE_COLS,
        "train_seasons": list(train_df["season"].unique()),
        "test_season": 2024
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved tuned RandomForest model → {MODEL_PATH}")

if __name__ == "__main__":
    main()
