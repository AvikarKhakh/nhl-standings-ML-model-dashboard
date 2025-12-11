import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
MASTER_PATH = DATA_DIR / "master_team_dataset.csv"
STANDINGS_PATH = DATA_DIR / "standings.csv"
OUTPUT_PATH = DATA_DIR / "master_team_with_points.csv"


def normalize_team_code(code: str) -> str:
    """
    Normalize team codes in the master dataset to NHL-style tricodes.

    You already aligned standings.csv to match master, but this keeps us safe
    in case there are weird variants like 'T.B' or 'S.J'.
    """
    if pd.isna(code):
        return code
    code = str(code).strip().upper()

    mapping = {
        "T.B": "TBL",
        "TAM": "TBL",
        "TB": "TBL",
        "S.J": "SJS",
        "SJS": "SJS",
        "SJ": "SJS",
        "N.J": "NJD",
        "NJ": "NJD",
        "MON": "MTL",
        "MTL": "MTL",
        "PHX": "ARI",  # old Coyotes code
        "ARI": "ARI",
        "LA": "LAK",
        "LAK": "LAK",
        "VGK": "VGK",
        "SEA": "SEA",
        "UTA": "UTA",
    }

    if code in mapping:
        return mapping[code]

    # default: already a tricode like BOS, NYR, etc.
    return code


def main():
    print(f"Reading master dataset from: {MASTER_PATH}")
    master = pd.read_csv(MASTER_PATH)

    # --- Normalize master team codes ---
    if "team" not in master.columns or "season" not in master.columns:
        raise ValueError(
            f"master_team_dataset.csv must contain 'team' and 'season' columns. "
            f"Found columns: {list(master.columns)}"
        )

    master["team"] = master["team"].astype(str).str.strip().apply(normalize_team_code)
    master["season"] = master["season"].astype(int)

    unique_codes = sorted(master["team"].unique())
    print("Unique team codes in master (after normalization):")
    print(unique_codes)

    # --- Read standings ---
    print(f"Reading standings from: {STANDINGS_PATH}")
    standings = pd.read_csv(STANDINGS_PATH)

    # Normalize standings column names to be robust
    standings.columns = standings.columns.str.strip()

    required_cols = ["season", "nhl_tricode", "team_name", "wins", "losses", "ot", "points"]
    missing = [c for c in required_cols if c not in standings.columns]
    if missing:
        raise ValueError(
            f"standings.csv is missing required columns: {missing}\n"
            f"Columns found: {list(standings.columns)}"
        )

    # Drop accidental header/invalid rows where 'season' is not numeric
    standings["season_num"] = pd.to_numeric(standings["season"], errors="coerce")
    before = len(standings)
    standings = standings.dropna(subset=["season_num"]).copy()
    dropped = before - len(standings)
    if dropped > 0:
        print(f"Dropped {dropped} accidental header/invalid rows from standings.")
    standings["season"] = standings["season_num"].astype(int)
    standings = standings.drop(columns=["season_num"])

    # ✅ FIXED: use .str.upper(), not .upper()
    standings["nhl_tricode"] = (
        standings["nhl_tricode"].astype(str).str.strip().str.upper()
    )

    # Make sure numeric
    for col in ["wins", "losses", "ot", "points"]:
        standings[col] = pd.to_numeric(standings[col], errors="coerce")

    # --- Merge ---
    print(
        "Merging master dataset with standings on "
        "master['season','team'] and standings['season','nhl_tricode']..."
    )

    merged = master.merge(
        standings[["season", "nhl_tricode", "wins", "losses", "ot", "points"]],
        left_on=["season", "team"],
        right_on=["season", "nhl_tricode"],
        how="left",
    )

    # Check for missing points
    missing_points = merged["points"].isna().sum()
    if missing_points > 0:
        print(
            f"WARNING: {missing_points} out of {len(merged)} rows have no points after the merge. "
            "This usually means a season/team vs nhl_tricode mismatch."
        )
        sample_missing = merged[merged["points"].isna()][["season", "team"]].head(20)
        print("Sample of season/team combinations with missing points:")
        print(sample_missing.to_string(index=False))
    else:
        print("All rows have points after merge. ✅")

    # Save output
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved merged dataset with points to: {OUTPUT_PATH}")
    print(f"Final shape: {merged.shape[0]} rows x {merged.shape[1]} columns")


if __name__ == "__main__":
    main()
