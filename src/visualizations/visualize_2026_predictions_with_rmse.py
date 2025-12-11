# src/visualize_2026_predictions_with_rmse.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PRED_PATH = Path("reports/predicted_points_2026.csv")  # <-- change if needed
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "predicted_points_2026_with_rmse_range.png"

# Use your held-out test season performance as a "plausible range" width
TEST_R2 = 0.341
TEST_RMSE = 13.96  # points

TEAM_COLORS = {
    "ANA": "#F47A38",
    "BOS": "#FFB81C",
    "BUF": "#002654",
    "CAR": "#CC0000",
    "CBJ": "#002654",
    "CGY": "#C8102E",
    "CHI": "#CF0A2C",
    "COL": "#6F263D",
    "DAL": "#006847",
    "DET": "#CE1126",
    "EDM": "#041E42",
    "FLA": "#041E42",
    "LAK": "#111111",
    "MIN": "#154734",
    "MTL": "#AF1E2D",
    "NJD": "#CE1126",
    "NSH": "#FFB81C",
    "NYI": "#00539B",
    "NYR": "#0038A8",
    "OTT": "#C52032",
    "PHI": "#F74902",
    "PIT": "#FFB81C",
    "SEA": "#001628",
    "SJS": "#006D75",
    "STL": "#002F87",
    "TBL": "#002868",
    "TOR": "#00205B",
    "UTA": "#8C2633",
    "VAN": "#00205B",
    "VGK": "#B4975A",
    "WPG": "#041E42",
    "WSH": "#041E42",
}

FALLBACK_COLOR = "#888888"


def main():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"Could not find predictions CSV at: {PRED_PATH}\n"
            f"Update PRED_PATH to wherever your file is saved."
        )

    df = pd.read_csv(PRED_PATH)

    # Accept either column name
    if "predicted_points_2026" not in df.columns:
        # try a common alternative
        if "predicted_points" in df.columns:
            df = df.rename(columns={"predicted_points": "predicted_points_2026"})
        else:
            raise ValueError(
                "CSV must have columns: team, predicted_points_2026 "
                "(or team, predicted_points)."
            )

    # Sort best -> worst so top of chart is best
    df = df.sort_values("predicted_points_2026", ascending=False).reset_index(drop=True)

    # Colors per team
    colors = [TEAM_COLORS.get(t, FALLBACK_COLOR) for t in df["team"]]

    # For plausible range labeling
    df["low"] = df["predicted_points_2026"] - TEST_RMSE
    df["high"] = df["predicted_points_2026"] + TEST_RMSE

    # =========================
    # Plot
    # =========================
    fig, ax = plt.subplots(figsize=(13, 11))

    # Main title (no overlap)
    fig.suptitle(
        "2025–26 Predicted NHL Points",
        fontsize=18,
        fontweight="bold",
        y=0.97,
        x=0.54,
    )

    # Subtitle on the axes
    ax.set_title(
        "Plausible range shown as ±RMSE from held-out 2024 season\n"
        f"Model context: Trained on 2013–2023 | Tested on 2024 | R² = {TEST_R2:.3f}, RMSE = {TEST_RMSE:.2f} pts",
        fontsize=11,
        pad=10,
    )

    # Bars
    bars = ax.barh(
        df["team"],
        df["predicted_points_2026"],
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
    )

    # Error bars (±RMSE)
    ax.errorbar(
        df["predicted_points_2026"],
        df["team"],
        xerr=TEST_RMSE,
        fmt="none",
        ecolor="black",
        elinewidth=2,
        capsize=3,
        alpha=0.85,
        zorder=3,
    )

    # Axes styling
    ax.set_xlabel("Projected Points (2025–26)", fontsize=12)
    ax.set_ylabel("")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()  # best team at top

    # Extra margin so team labels fit nicely
    plt.subplots_adjust(left=0.18, top=0.88)

    # Make x-limits wide enough to show error bars + labels
    x_min = max(0, df["low"].min() - 5)
    x_max = df["high"].max() + 8
    ax.set_xlim(x_min, x_max)

    # Save
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
