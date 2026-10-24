from pathlib import Path

import pandas as pd

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV file found in the data folder. "
        "Please place your employee dataset inside data/."
    )

# Load the first CSV file
dataset_path = csv_files[0]
df = pd.read_csv(dataset_path)

print("\n========== DATASET DETAILS ==========")
print(f"Dataset file: {dataset_path.name}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)
