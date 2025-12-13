from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path("data/master_team_with_points.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Same training window as the tuned model
TRAIN_SEASONS = list(range(2013, 2024))  # 2013–2023


def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Ensure season is integer
    df["season"] = df["season"].astype(int)

    # Use only training seasons (so we don't "peek" at the test year)
    df_train = df[df["season"].isin(TRAIN_SEASONS)].copy()
    print(
        f"Using seasons for correlation heatmap: "
        f"{sorted(df_train['season'].unique())}"
    )

    # Feature set aligned with tuned Random Forest + target
    feature_cols = [
        "xgf60",
        "xga60",
        "ev_xgf60",
        "ev_xga60",
        "pp_xgf60",
        "pp_xga60",
        "pk_xgf60",
        "pk_xga60",
        "hdf",
        "hda",
        "hdf_percent",
        "xgoals_percentage",
        "corsi_percentage",
        "fenwick_percentage",
        "penalties_drawn_per60",
        "penalties_taken_per60",
        "penalty_diff",
        "gsa_per60",
        "save_percent",
        "hd_save_percent",
        "points",  # target
    ]

    missing = [c for c in feature_cols if c not in df_train.columns]
    if missing:
        raise ValueError(f"Missing expected columns in dataset: {missing}")

    corr = df_train[feature_cols].corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        annot=False,
        cbar_kws={"label": "Correlation (Pearson r)"},
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title("Correlation Heatmap – NHL Microstats vs Points (2013–2023)")
    plt.tight_layout()

    out_path = REPORTS_DIR / "correlation_heatmap_points.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to: {out_path}")


if __name__ == "__main__":
    main()
