import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/master_team_with_points.csv")
MODEL_PATH = Path("models/points_rf.pkl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "feature_importance_points.png"


def main():
    print(f"Loading model bundle from: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load data to recover feature columns in the same way as train_points_model.py
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["season"] = df["season"].astype(int)

    # Must match the exclusions from train_points_model.py
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

    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Loaded model does not have feature_importances_ attribute.")

    importances = model.feature_importances_
    if len(importances) != len(feature_cols):
        raise ValueError(
            f"Length mismatch: {len(importances)} importances vs {len(feature_cols)} features"
        )

    # Sort by importance
    indices = np.argsort(importances)
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances for Predicting NHL Points")
    plt.tight_layout()

    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"\nSaved feature importance plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
