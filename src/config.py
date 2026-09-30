"""Project-wide settings. Every other module imports paths and constants from here."""
from pathlib import Path
import os

from dotenv import load_dotenv

# Project root = the regime-portfolio/ folder, wherever code is run from
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
MODEL_DIR = ROOT / "model"
REPORTS_DIR = ROOT / "reports"

FRED_API_KEY = os.getenv("FRED_API_KEY")  # not needed yet; used if we add macro features

RANDOM_SEED = 42
