"""Settings for the tabular_automl package"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

LARGE_DATASET_ROWS = int(1e5)

SUPPORTED_TASK_TYPES = ("regression", "classification")

FILE_READERS = {".csv": pd.read_csv}
