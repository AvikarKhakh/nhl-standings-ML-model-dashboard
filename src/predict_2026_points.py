import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATH = Path("models/points_rf.pkl")
INPUT_PATH = Path("data/master_team_2025_features.csv")   # <-- you create this
OUTPUT_PATH = Path("reports/predicted_points_2026.csv")

# Must match training-time feature set EXACTLY (names + order)
FEATURE_COLS = [
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

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input features file not found: {INPUT_PATH}")

    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    print(f"Loading features: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    # Basic checks
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required feature columns: {missing}")

    # Keep identifiers if you have them (team/nhl_tricode), but don't feed into model
    id_cols = [c for c in ["team", "nhl_tricode"] if c in df.columns]
    X = df[FEATURE_COLS].values.astype(float)

    preds = model.predict(X)

    out = df[id_cols].copy() if id_cols else pd.DataFrame()
    out["predicted_points_2026"] = preds

    # Optional: sort high to low
    out = out.sort_values("predicted_points_2026", ascending=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved predictions to: {OUTPUT_PATH}")
    print(out.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
