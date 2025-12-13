import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

DATA_PATH = Path("data/master_team_with_points.csv")
MODEL_PATH = Path("models/points_rf.pkl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = list(range(2013, 2024))  # 2013–2023
TEST_SEASON = 2024                       # held-out season


def load_data_and_features():
    """Load dataset and build the same feature set used in train_points_model.py."""
    print(f"Loading merged dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Ensure correct types
    df["season"] = df["season"].astype(int)

    # Same exclude set as train_points_model.py
    exclude_cols = {
        "season",
        "team",
        "points",
        "nhl_tricode",
        "wins",
        "losses",
        "ot",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    print("\nFeature columns (order used by the model):")
    for col in feature_cols:
        print(f"  - {col}")

    # Split train/test by season
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

    return df, train_df, test_df, X_train, y_train, X_test, y_test, feature_cols


def load_model():
    """Load the tuned RandomForestRegressor saved by train_points_model.py."""
    print(f"Loading model from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def plot_scatter(test_df, y_test, y_pred):
    """Scatter plot: actual vs predicted points for the test season."""
    teams = test_df["team"].values 

    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, s=60)

    # 45-degree reference line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)

    # Annotate each point with team code
    for actual, pred, code in zip(y_test, y_pred, teams):
        plt.text(actual + 0.3, pred + 0.3, code, fontsize=8)

    plt.xlabel("Actual Points (2024)")
    plt.ylabel("Predicted Points (2024)")
    plt.title("NHL Team Points – Actual vs Predicted (2024, Tuned Random Forest)")
    plt.grid(alpha=0.3)

    out_path = REPORTS_DIR / "points_actual_vs_predicted_2024.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved scatter plot to: {out_path}")


def plot_residual_bars(test_df, y_test, y_pred):
    """
    Bar chart of residuals (predicted − actual) per team, sorted.
    Positive = model over-predicted, negative = under-predicted.
    """
    residuals = y_pred - y_test
    result_df = pd.DataFrame({
        "team": test_df["team"].values,
        "actual": y_test,
        "predicted": y_pred,
        "residual": residuals,
    })

    # Sort by residual (under-prediction on the left, over-prediction on the right)
    result_df = result_df.sort_values("residual")

    plt.figure(figsize=(12, 6))
    bars = plt.bar(result_df["team"], result_df["residual"])

    # Color: under-prediction one color, over-prediction another
    for bar, val in zip(bars, result_df["residual"]):
        if val < 0:
            bar.set_color("#d62728")  # red-ish for under
        else:
            bar.set_color("#1f77b4")  # blue-ish for over

    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Predicted − Actual Points")
    plt.xlabel("Team (2024)")
    plt.title("Prediction Error by Team – 2024 Season (Tuned Random Forest)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    out_path = REPORTS_DIR / "points_residuals_bar_2024.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved residual bar chart to: {out_path}")


def main():
    df, train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = (
        load_data_and_features()
    )
    model = load_model()

    # Evaluate on train & test for context
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))

    print("=== Tuned Model Performance (re-evaluated) ===")
    print(f"Train R^2:  {train_r2:.3f}")
    print(f"Train RMSE: {train_rmse:.3f}")
    print(f"Test R^2:   {test_r2:.3f}")
    print(f"Test RMSE:  {test_rmse:.3f}\n")

    # Visualizations
    plot_scatter(test_df, y_test, y_test_pred)
    plot_residual_bars(test_df, y_test, y_test_pred)


if __name__ == "__main__":
    main()
