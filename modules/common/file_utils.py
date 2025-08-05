import os
import pandas as pd
from typing import Dict

def save_csv_data(df: pd.DataFrame, metadata: Dict[str, str], output_path: str, filename: str) -> str:
    """
    Saves a DataFrame to a CSV file in a structured directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        metadata (Dict[str, str]): A dictionary containing metadata like project, date, time, location.
        output_path (str): The root directory to save the output CSV file.
        filename (str): The desired filename.

    Returns:
        str: The full path to the generated CSV file, or None if an error occurs.
    """
    try:
        project, date, cycle, location = metadata["project"], metadata["date_str"], metadata["cycle"], metadata["location"]
        
        output_dir = os.path.join(output_path, project, date, cycle, location)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.splitext(filename)[1]:
            filename += ".csv"

        output_csv_path = os.path.join(output_dir, filename)

        df.to_csv(output_csv_path, index=False)
        print(f"[SAVING]: Successfully saved data to {output_csv_path}")
        return output_csv_path
    except Exception as e:
        print(f"[SAVING]: Error saving CSV file: {e}")
        return None 