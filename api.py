from datetime import datetime
from pathlib import Path
import subprocess
import sys
import json

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your React frontend (Vite) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache so refresh doesn't re-run the model
LATEST = {
    "standings": [],
    "metrics": {"last_run": None, "rmse": None, "r2": None, "n_samples": None},
}


TEAM_TO_DIVISION = {
    # Atlantic
    "BOS": "Atlantic", "BUF": "Atlantic", "DET": "Atlantic", "FLA": "Atlantic",
    "MTL": "Atlantic", "OTT": "Atlantic", "TBL": "Atlantic", "TOR": "Atlantic",

    # Metropolitan
    "CAR": "Metro", "CBJ": "Metro", "NJD": "Metro", "NYI": "Metro",
    "NYR": "Metro", "PHI": "Metro", "PIT": "Metro", "WSH": "Metro",

    # Central
    "CHI": "Central", "COL": "Central", "DAL": "Central", "MIN": "Central",
    "NSH": "Central", "STL": "Central", "UTA": "Central", "WPG": "Central",

    # Pacific
    "ANA": "Pacific", "CGY": "Pacific", "EDM": "Pacific", "LAK": "Pacific",
    "SJS": "Pacific", "SEA": "Pacific", "VAN": "Pacific", "VGK": "Pacific",
}

DIVISION_TO_CONFERENCE = {
    "Atlantic": "ECF",
    "Metro": "ECF",
    "Central": "WCF",
    "Pacific": "WCF",
}

def _division_and_conference(team: str) -> tuple[str, str]:
    code = (team or "").strip().upper()
    division = TEAM_TO_DIVISION.get(code, "—")
    conference = DIVISION_TO_CONFERENCE.get(division, "—")
    return division, conference

def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _run_predict_script() -> tuple[int, str, str]:
    """
    Runs your existing prediction script using the current venv python.
    Returns (returncode, stdout, stderr).
    """
    root = _project_root()
    script_path = root / "src" / "predict_2026_points.py"

    if not script_path.exists():
        return (1, "", f"Script not found: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    return (result.returncode, result.stdout, result.stderr)


def _load_predictions_csv() -> pd.DataFrame:
    """
    Loads the CSV your project already writes to:
    reports/predicted_points_2026.csv
    """
    root = _project_root()
    csv_path = root / "reports" / "predicted_points_2026.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at: {csv_path}")

    return pd.read_csv(csv_path)


def _load_metrics_json() -> dict:
    """
    Loads metrics written by src/train_points_model.py:
    reports/metrics_points.json
    """
    root = _project_root()
    metrics_path = root / "reports" / "metrics_points.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r") as f:
        return json.load(f)


def _detect_team_points_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    Attempts to find the team column and predicted points column
    using common column name patterns.
    """
    cols = list(df.columns)

    # Find team column
    team_col = None
    for c in cols:
        lc = c.lower()
        if "team" in lc or "abbrev" in lc:
            team_col = c
            break

    # Find points column
    points_col = None
    for c in cols:
        lc = c.lower()
        if "predicted_points" in lc:
            points_col = c
            break
    if points_col is None:
        # fallback: something like "Predicted Points" or "points_pred" or "points"
        for c in cols:
            lc = c.lower()
            if ("point" in lc and "pred" in lc) or lc == "points":
                points_col = c
                break

    if team_col is None or points_col is None:
        raise ValueError(f"Could not detect team/points columns. Columns found: {cols}")

    return team_col, points_col


def _to_standings_json(df: pd.DataFrame) -> list[dict]:
    """
    Converts DF into the JSON structure the frontend expects:
    rank, team, points, conference, division
    """
    team_col, points_col = _detect_team_points_cols(df)

    df_sorted = df.sort_values(by=points_col, ascending=False).reset_index(drop=True)
    df_sorted["rank"] = df_sorted.index + 1

    standings = []
    for _, row in df_sorted.iterrows():
        team_code = str(row[team_col]).strip().upper()
        division, conference = _division_and_conference(team_code)

        standings.append(
            {
                "rank": int(row["rank"]),
                "team": team_code,
                "points": round(float(row[points_col]), 2),
                "conference": conference,
                "division": division,
            }
        )

    return standings

@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/run")
def run_model():
    """
    1) Runs src/predict_2026_points.py
    2) Loads reports/predicted_points_2026.csv
    3) Loads reports/metrics_points.json
    4) Returns standings + metrics JSON to the frontend
    """
    code, out, err = _run_predict_script()

    if code != 0:
        return {
            "error": "predict_2026_points.py failed",
            "returncode": code,
            "stderr": err[-2000:],
            "stdout": out[-2000:],
        }

    try:
        df = _load_predictions_csv()
        standings = _to_standings_json(df)
    except Exception as e:
        return {"error": "Failed to load/parse predictions CSV", "details": str(e)}

    file_metrics = _load_metrics_json()

    LATEST["standings"] = standings
    LATEST["metrics"] = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmse": file_metrics.get("rmse"),
        "r2": file_metrics.get("r2"),
        "n_samples": file_metrics.get("n_samples"),
    }

    return {"standings": standings, "metrics": LATEST["metrics"]}


@app.get("/api/results/latest")
def latest_results():
    return {"standings": LATEST["standings"]}


@app.get("/api/metrics")
def metrics():
    return LATEST["metrics"]
