import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If SHAP isn't installed yet:
# pip install shap
import shap


DATA_PATH = Path("data/master_team_with_points.csv")
MODEL_PATH = Path("models/points_rf.pkl")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SEASONS = list(range(2013, 2024))  # 2013–2023
TARGET_COL = "points"

# 4 features you want to explain
FEATURES_TO_PLOT = ["xgf60", "xga60", "xgoals_percentage", "save_percent"]

# Match your model training exclusions
EXCLUDE_COLS = {
    "season", "team", "points", "nhl_tricode",
    "wins", "losses", "ot"
}


def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_feature_cols(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def prettify_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, linestyle="--", linewidth=1)  # baseline SHAP=0
    ax.tick_params(axis="both", labelsize=10)


def main():
    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["season"] = df["season"].astype(int)

    # Use training seasons for SHAP explanations (consistent with model training)
    df_train = df[df["season"].isin(TRAIN_SEASONS)].copy()

    feature_cols = get_feature_cols(df_train)

    # Sanity: ensure requested features exist
    missing_feats = [f for f in FEATURES_TO_PLOT if f not in feature_cols]
    if missing_feats:
        raise ValueError(f"Missing required features in dataset/model features: {missing_feats}")

    X = df_train[feature_cols]
    print(f"Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # Subsample for speed + cleaner plot (optional)
    # If you want full: comment this out
    if len(X) > 350:
        X_plot = X.sample(n=350, random_state=42)
    else:
        X_plot = X

    # SHAP values
    print("Computing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_plot)

    # shap_values is (n_samples, n_features) for regression
    shap_matrix = np.array(shap_values)

    # Map feature -> shap column index
    feat_to_idx = {feat: i for i, feat in enumerate(feature_cols)}

    # ---------- Plot dashboard ----------
    plt.figure(figsize=(14, 10))
    plt.suptitle(
        "SHAP Dependence Dashboard — Key Drivers of Predicted NHL Points\n"
        "(Model trained on 2013–2023 seasons)",
        fontsize=16,
        y=0.98
    )

    axes = []
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        axes.append(ax)

    for ax, feat in zip(axes, FEATURES_TO_PLOT):
        idx = feat_to_idx[feat]
        x = X_plot[feat].values
        y = shap_matrix[:, idx]

        # Scatter
        ax.scatter(x, y, alpha=0.75)

        # Titles + labels
        ax.set_title(f"{feat} → impact on predicted points", fontsize=13)
        prettify_axes(ax, xlabel=feat, ylabel=f"SHAP value for {feat}")

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = REPORTS_DIR / "shap_dependence_dashboard.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved SHAP dashboard to: {out_path}")


if __name__ == "__main__":
    main()
