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

def _parse_hhmmssmmm(time_str: str) -> datetime.time:
    """Parse a HHMMSSmmm time string into a datetime.time value."""
    if len(time_str) != 9:
        raise ValueError(
            f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm' format."
        )
    time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
    return datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

def parse_file_metadata(directory_path: Path) -> Dict:
    """
    Parses an image filename to get the replicate number, recording time, and date.
    """
    filenames = _list_image_filenames(directory_path)
    raw_img_paths = [str(directory_path / filename) for filename in filenames]
    
    # Get the last image file in the list (filenames are sorted)
    image_filename = filenames[-1]

    try:
        base_filename = Path(image_filename).stem
        filename_parts = base_filename.split("_")

        if len(filename_parts) < 3:
            raise ValueError(f"Filename '{image_filename}' does not have enough parts separated by '_'. Expected format: '..._YYYYMMDD_HHMMSSmmm_replicate.ext'")

        # Extract time and date from the last image (which has the highest replicate number)
        time_str = filename_parts[-2]
        date_from_filename_str = filename_parts[-3]
        recording_start_date = datetime.datetime.strptime(date_from_filename_str, "%Y%m%d").date()
        recording_start_time = _parse_hhmmssmmm(time_str)

        # Total replicates is simply the number of image files in the directory
        total_replicates = len(filenames)

    except (IndexError, ValueError) as e:
        if isinstance(e, ValueError) and ("incorrect length" in str(e) or "enough parts" in str(e)):
            raise
        raise ValueError(f"Could not parse info from filename '{image_filename}'. Expected '..._YYYYMMDD_HHMMSSmmm_replicate.ext' format. Error: {e}")
    
    return {
        "total_replicates": total_replicates,
        "recording_start_time": recording_start_time,
        "recording_start_date": recording_start_date,
        "raw_img_paths": raw_img_paths
    }