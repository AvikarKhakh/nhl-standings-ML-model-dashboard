# src/visualize_trends.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Paths
DATA_PATH = Path("data/master_team_with_points.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = REPORTS_DIR / "trendlines_core_features.png"


def main():
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Ensure season is numeric
    df["season"] = df["season"].astype(int)

    # Core features we care about for trends
    core_features = [
        "xgf60",
        "xga60",
        "xgoals_percentage",
        "save_percent",
    ]

    # Sanity check: make sure all features exist
    missing = [f for f in core_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features in dataset: {missing}")

    # Compute league-average for each season and feature
    season_means = (
        df.groupby("season")[core_features]
        .mean()
        .reset_index()
        .sort_values("season")
    )

    seasons = season_means["season"].values

    print("Seasons in dataset:", seasons)

    # COVID-shortened seasons (2019–20 and 2020–21),
    # which correspond to seasons labeled 2020 and 2021 in our data.
    covid_seasons = [2020, 2021]

    # Plot setup – 2x2 grid for 4 core features
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    titles = {
        "xgf60": "Expected Goals For per 60 (xGF/60)",
        "xga60": "Expected Goals Against per 60 (xGA/60)",
        "xgoals_percentage": "Expected Goals Share (xG%)",
        "save_percent": "Save Percentage",
    }

    for ax, feature in zip(axes, core_features):
        y_vals = season_means[feature].values

        ax.plot(
            seasons,
            y_vals,
            marker="o",
            linewidth=2,
        )

        # Vertical red lines for COVID seasons
        for cs in covid_seasons:
            if cs in seasons:
                ax.axvline(
                    cs,
                    color="red",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                )

        ax.set_title(titles.get(feature, feature))
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3)

    # Shared x-label and ticks
    for ax in axes:
        ax.set_xticks(seasons)
        ax.set_xlabel("Season")

    fig.suptitle(
        "League-Average Microstat Trends Over Time\n"
        "(Red dashed lines = COVID-shortened seasons 2019–20 and 2020–21)",
        fontsize=14,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved core feature trendlines to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()