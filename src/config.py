# config.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Environment
ENV = os.getenv("ENV", "local")

# Storage paths (local or S3-ready)
RAW_DATA_PATH = os.getenv(
    "RAW_DATA_PATH",
    str(BASE_DIR / "data" / "raw" / "telco_churn.csv")
)

STAGING_PATH = os.getenv(
    "STAGING_PATH",
    str(BASE_DIR / "data" / "staging")
)

PROCESSED_PATH = os.getenv(
    "PROCESSED_PATH",
    str(BASE_DIR / "data" / "processed")
)

MODEL_OUTPUT_PATH = os.getenv(
    "MODEL_OUTPUT_PATH",
    str(BASE_DIR / "models")
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)