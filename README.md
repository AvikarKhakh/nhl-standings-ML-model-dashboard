# NHL Standings Machine Learning

This repository contains a machine learning project designed to predict NHL team standings for the 2026 season. It includes a backend for data processing and prediction, as well as a frontend dashboard for visualizing the results.

---

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Running the Frontend](#running-the-frontend)
- [Connecting Backend to Dashboard](#connecting-backend-to-dashboard)
- [Main Run File](#main-run-file)
- [Contributing](#contributing)
- [License](#license)

---

## Getting Started

Follow the instructions below to set up and run the project on your local machine.

---
## Prerequisites

Ensure you have the following installed:

- **Python 3.8 or higher**: Required for running the backend and machine learning scripts.
- **pip (Python package manager)**: For installing Python dependencies.
- **Node.js and npm**: Needed for setting up and running the frontend dashboard.
- **Git**: To clone the repository.
- **uvicorn**: For running the FastAPI backend server. Install it using:
- **fastapi**: Required for the backend API. Install it using:
- **Any additional Python dependencies**: Listed in `requirements.txt`. Install them using:

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/nhl-standings-ml.git
    cd nhl-standings-ml
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the `frontend` directory and install dependencies:
    ```bash
    cd nhl-ml-dashboard
    npm install
    ```

---

## Usage

To generate predictions for the 2026 NHL standings, run the main Python script in directory `nhl_standings_ml`:
```bash
python src/predict_2026_points.py
```

This script processes the data, trains the model, and outputs the predicted standings.

---

## Running the Frontend

1. Navigate to the `frontend` directory:
    ```bash
    cd nhl-ml-dashboard
    ```

2. Start the development server:
    ```bash
    npm run dev
    ```

3. Open your browser and go to `http://localhost:5173` to view the dashboard.

---

## Connecting Backend to Dashboard

To connect the backend to the dashboard, follow these steps:

1. **Install Required Packages**  
    Ensure you have `uvicorn` and `fastapi` installed. If not, install them using pip:
    ```bash
    pip install uvicorn fastapi
    ```

2. **Run the Backend**  
    Navigate to the `nhl_standings_ml` directory and start the backend server:
    ```bash
    uvicorn api:app --reload --port 8000
    ```

3. **Update Frontend Configuration**  
    The backend will expose an API endpoint (e.g., `http://localhost:8000`). Update the frontend configuration file (e.g., `frontend/src/config.js`) to point to this endpoint:
    ```javascript
    export const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";
    ```

4. **Restart the Frontend Server**  
    After updating the configuration, restart the frontend server:
    ```bash
    npm run dev
    ```

Once completed, the backend and frontend should be connected, and the dashboard will display the predictions.

---

## Main Run File

The main file for generating predictions is:
```bash
predict_2026_points.py
```

This script produces the results used by the dashboard.

---
