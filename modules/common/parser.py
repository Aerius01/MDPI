import datetime
import os
from pathlib import Path
from typing import Dict

# Shared helpers and constants
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff'}


def _list_image_filenames(directory_path: Path) -> list[str]:
    """Return sorted image filenames in a directory filtered by IMAGE_EXTENSIONS."""
    filenames = [
        f for f in os.listdir(directory_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    if not filenames:
        raise ValueError(f"No image files found in directory '{directory_path}'")
    filenames.sort()
    return filenames


def _build_image_paths(directory_path: Path, filenames: list[str]) -> list[str]:
    """Build full paths for the given filenames within the directory path."""
    return [str(directory_path / filename) for filename in filenames]


def _parse_hhmmssmmm(time_str: str) -> datetime.time:
    """Parse a HHMMSSmmm time string into a datetime.time value."""
    if len(time_str) != 9:
        raise ValueError(
            f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm' format."
        )
    time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
    return datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

def _parse_path_metadata(directory_path: Path) -> Dict:
    """Parses the directory path to extract project, date, cycle, and location."""
    try:
        location = directory_path.name
        cycle = directory_path.parent.name
        date_str = directory_path.parent.parent.name
        project = directory_path.parent.parent.parent.name
    except IndexError:
        raise ValueError(f"Directory path '{directory_path}' is not in the expected format '.../project/date/cycle/location'")

    if not date_str:
        raise ValueError(f"Extracted date is empty. Please check the directory path '{directory_path}' format.")

    try:
        recording_start_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    except ValueError:
        raise ValueError(f"Date '{date_str}' from path is not in 'YYYYMMDD' format.")
    
    return {
        "project": project,
        "recording_start_date": recording_start_date,
        "cycle": cycle,
        "location": location,
    }

def _parse_file_metadata(directory_path: Path, recording_start_date: datetime.date) -> Dict:
    """Parses an image filename to get the replicate number and recording time."""
    filenames = _list_image_filenames(directory_path)
    raw_img_paths = _build_image_paths(directory_path, filenames)
    
    # Get the last image file in the list (filenames are sorted)
    image_filename = filenames[-1]

    try:
        base_filename = Path(image_filename).stem
        filename_parts = base_filename.split("_")

        if len(filename_parts) < 3:
            raise ValueError(f"Filename '{image_filename}' does not have enough parts separated by '_'. Expected format: '..._YYYYMMDD_HHMMSSmmm_replicate.ext'")

        total_replicates = int(filename_parts[-1])
        time_str = filename_parts[-2]
        date_from_filename_str = filename_parts[-3]
        date_from_filename = datetime.datetime.strptime(date_from_filename_str, "%Y%m%d").date()

        # Check if the date in the filename matches the date in the path
        if date_from_filename != recording_start_date:
            raise ValueError(f"Date in filename '{date_from_filename}' does not match date in path '{recording_start_date}'.")

        recording_start_time = _parse_hhmmssmmm(time_str)

    except (IndexError, ValueError) as e:
        if isinstance(e, ValueError) and ("does not match" in str(e) or "incorrect length" in str(e) or "enough parts" in str(e)):
            raise
        raise ValueError(f"Could not parse info from filename '{image_filename}'. Expected '..._YYYYMMDD_HHMMSSmmm_replicate.ext' format. Error: {e}")
    
    return {
        "total_replicates": total_replicates,
        "recording_start_time": recording_start_time,
        "raw_img_paths": raw_img_paths
    }

def parse_metadata(directory_path: Path) -> Dict:
    """Parses directory path and image filenames to extract all metadata."""
    path_metadata = _parse_path_metadata(directory_path)
    file_metadata = _parse_file_metadata(directory_path, path_metadata["recording_start_date"])

    return {**path_metadata, **file_metadata}