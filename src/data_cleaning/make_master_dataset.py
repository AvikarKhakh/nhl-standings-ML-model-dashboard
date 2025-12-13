import numpy as np
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_CSV = DATA_DIR / "master_team_dataset.csv"

FINAL_COLUMNS = [
    "team",
    "season",
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
]


def load_team_season(path: Path) -> pd.DataFrame:
    """Load a single teams_YYYY_YY.csv and compute per-60 features for each team."""
    print(f"Loading {path}")
    df = pd.read_csv(path)

    # Only keep team-level rows
    if "position" in df.columns:
        df = df[df["position"] == "Team Level"].copy()

    # We expect a 'season' column already in these files
    if "season" not in df.columns:
        raise ValueError(f"'season' column not found in {path.name}")

    # Helper to compute per-60 safely
    def per60(series, seconds):
        seconds = seconds.astype(float).replace(0, np.nan)
        return series * 3600.0 / seconds

    all_df = df[df["situation"] == "all"].copy()
    if all_df.empty:
        raise ValueError(f"No 'all' situation rows in {path.name}")

    out = pd.DataFrame()
    out["team"] = all_df["team"].astype(str)
    out["season"] = all_df["season"].astype(int)

    sec_all = all_df["iceTime"].astype(float).replace(0, np.nan)

    # core xG per 60
    out["xgf60"] = per60(all_df["xGoalsFor"], sec_all)
    out["xga60"] = per60(all_df["xGoalsAgainst"], sec_all)

    # xG share, Corsi, Fenwick
    out["xgoals_percentage"] = all_df.get("xGoalsPercentage", np.nan)
    out["corsi_percentage"] = all_df.get("corsiPercentage", np.nan)
    out["fenwick_percentage"] = all_df.get("fenwickPercentage", np.nan)

    # high-danger xG
    out["hdf"] = all_df.get("highDangerxGoalsFor", np.nan)
    out["hda"] = all_df.get("highDangerxGoalsAgainst", np.nan)
    denom_hd = out["hdf"] + out["hda"]
    out["hdf_percent"] = np.where(denom_hd > 0, out["hdf"] / denom_hd, np.nan)

    # penalties per 60
    pen_for = all_df.get("penaltiesFor", np.nan)
    pen_against = all_df.get("penaltiesAgainst", np.nan)
    out["penalties_drawn_per60"] = per60(pen_for, sec_all)
    out["penalties_taken_per60"] = per60(pen_against, sec_all)
    out["penalty_diff"] = out["penalties_drawn_per60"] - out["penalties_taken_per60"]

    # goaltending: GSAx per 60 (xGA - GA), save%, high-danger save%
    xga = all_df.get("xGoalsAgainst", np.nan)
    ga = all_df.get("goalsAgainst", np.nan)
    out["gsa_per60"] = per60(xga - ga, sec_all)

    shots_against = all_df.get("shotsOnGoalAgainst", np.nan)
    out["save_percent"] = np.where(
        (shots_against > 0) & (~shots_against.isna()) & (~ga.isna()),
        1.0 - ga / shots_against,
        np.nan,
    )

    hd_shots_against = all_df.get("highDangerShotsAgainst", np.nan)
    hd_goals_against = all_df.get("highDangerGoalsAgainst", np.nan)
    out["hd_save_percent"] = np.where(
        (hd_shots_against > 0) & (~hd_shots_against.isna()) & (~hd_goals_against.isna()),
        1.0 - hd_goals_against / hd_shots_against,
        np.nan,
    )

    ev_df = df[df["situation"] == "5on5"].copy()
    if not ev_df.empty:
        sec_ev = ev_df["iceTime"].astype(float).replace(0, np.nan)
        ev_features = pd.DataFrame()
        ev_features["team"] = ev_df["team"].astype(str)
        ev_features["season"] = ev_df["season"].astype(int)
        ev_features["ev_xgf60"] = per60(ev_df["xGoalsFor"], sec_ev)
        ev_features["ev_xga60"] = per60(ev_df["xGoalsAgainst"], sec_ev)

        out = out.merge(ev_features, on=["team", "season"], how="left")
    else:
        out["ev_xgf60"] = np.nan
        out["ev_xga60"] = np.nan

    pp_df = df[df["situation"] == "5on4"].copy()
    if not pp_df.empty:
        sec_pp = pp_df["iceTime"].astype(float).replace(0, np.nan)
        pp_features = pd.DataFrame()
        pp_features["team"] = pp_df["team"].astype(str)
        pp_features["season"] = pp_df["season"].astype(int)
        pp_features["pp_xgf60"] = per60(pp_df["xGoalsFor"], sec_pp)
        pp_features["pp_xga60"] = per60(pp_df["xGoalsAgainst"], sec_pp)
        out = out.merge(pp_features, on=["team", "season"], how="left")
    else:
        out["pp_xgf60"] = np.nan
        out["pp_xga60"] = np.nan

    pk_df = df[df["situation"] == "4on5"].copy()
    if not pk_df.empty:
        sec_pk = pk_df["iceTime"].astype(float).replace(0, np.nan)
        pk_features = pd.DataFrame()
        pk_features["team"] = pk_df["team"].astype(str)
        pk_features["season"] = pk_df["season"].astype(int)
        pk_features["pk_xgf60"] = per60(pk_df["xGoalsFor"], sec_pk)
        pk_features["pk_xga60"] = per60(pk_df["xGoalsAgainst"], sec_pk)
        out = out.merge(pk_features, on=["team", "season"], how="left")
    else:
        out["pk_xgf60"] = np.nan
        out["pk_xga60"] = np.nan

    # Ensure all expected columns exist
    for col in FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[FINAL_COLUMNS]


def main():
    team_files = sorted(DATA_DIR.rglob("teams_*.csv"))

    if not team_files:
        raise FileNotFoundError(f"No team CSVs found under {DATA_DIR}")

    frames = []
    for path in team_files:
        try:
            df_season = load_team_season(path)
            frames.append(df_season)
        except Exception as e:
            print(f"⚠️ Skipping {path.name} due to error: {e}")

    if not frames:
        raise RuntimeError("No valid team datasets were loaded.")

    master = pd.concat(frames, ignore_index=True)

    # Basic sanity checks
    print("Master dataset shape:", master.shape)
    print("Seasons in dataset:", sorted(master["season"].unique()))
    print("Example rows:")
    print(master.head())

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved master dataset to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()