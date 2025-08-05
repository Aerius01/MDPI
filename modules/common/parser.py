import datetime
import os
from pathlib import Path
from typing import Dict, List

def parse_path_metadata(directory_path: Path) -> Dict:
    """Parses the directory path to extract project, date, cycle, and location."""
    try:
        p = directory_path
        location = p.name
        cycle = p.parent.name
        date_str = p.parent.parent.name
        project = p.parent.parent.parent.name
    except IndexError:
        raise ValueError(f"Directory path '{directory_path}' is not in the expected format '.../project/date/cycle/location'")

    try:
        recording_start_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"Date '{date_str}' from path is not in 'YYYYMMDD' format.")
    
    return {
        "project": project,
        "recording_start_date": recording_start_date,
        "cycle": cycle,
        "location": location,
        "date_str": date_str
    }

def parse_file_metadata(directory_path: Path, recording_start_date: datetime.date) -> Dict:
    """Parses an image filename to get the replicate number and recording time."""
    p = directory_path
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    filenames = sorted([f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in image_extensions])
    if not filenames:
        raise ValueError(f"No image files found in directory '{p}'")

    raw_imgs = [str(p / filename) for filename in filenames]
    image_filename = filenames[0]

    try:
        base_filename, _ = os.path.splitext(image_filename)
        filename_parts = base_filename.split("_")

        if len(filename_parts) < 4:
            raise ValueError("Filename does not have enough parts separated by '_'")

        replicate = int(filename_parts[-1])
        date_from_filename_str = filename_parts[-3]
        time_str = filename_parts[-2]

        date_from_filename = datetime.datetime.strptime(date_from_filename_str, "%Y%m%d").date()

        if date_from_filename != recording_start_date:
            raise ValueError(f"Date in filename '{date_from_filename}' does not match date in path '{recording_start_date}'.")

        if len(time_str) != 9:
            raise ValueError(f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm'.")

        time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
        recording_start_time = datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

    except (IndexError, ValueError) as e:
        if isinstance(e, ValueError) and ("does not match" in str(e) or "incorrect length" in str(e) or "enough parts" in str(e)):
            raise
        raise ValueError(f"Could not parse info from filename '{image_filename}'. Expected '..._YYYYMMDD_HHMMSSmmm_replicate.ext' format. Error: {e}")
    
    return {
        "replicate": replicate,
        "recording_start_time": recording_start_time,
        "raw_imgs": raw_imgs
    }

def find_single_csv_file(directory_path: Path) -> str:
    """Finds the single .csv file in the directory and returns its path."""
    p = directory_path
    csv_files = [f for f in os.listdir(p) if f.lower().endswith('.csv')]

    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in directory '{p}'")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory '{p}': {', '.join(csv_files)}")

    return str(p / csv_files[0]) 