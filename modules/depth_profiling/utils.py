from pathlib import Path
import os


def find_single_csv_file(directory_path: Path) -> str:
    """Finds the single .csv file in the directory and returns its path."""
    p = directory_path
    csv_files = [f for f in os.listdir(p) if f.lower().endswith('.csv')]

    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in directory '{p}'")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory '{p}': {', '.join(csv_files)}")

    return str(p / csv_files[0])


