import datetime
import os
from pathlib import Path
from typing import Dict
import cv2

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

def find_single_csv_file(directory_path: Path) -> str:
    """Finds the single .csv file in the directory and returns its path."""
    p = directory_path
    csv_files = [f for f in os.listdir(p) if f.lower().endswith('.csv')]

    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in directory '{p}'")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory '{p}': {', '.join(csv_files)}")

    return str(p / csv_files[0])

def parse_file_metadata(directory_path: Path) -> Dict:
    """
    Parses an image filename to get the replicate number, recording time, and date.
    """
    filenames = _list_image_filenames(directory_path)
    raw_img_paths = [str(directory_path / filename) for filename in filenames]
    
    # Get image resolution from the first image
    first_image_path = raw_img_paths[0]
    image = cv2.imread(first_image_path)
    if image is None:
        raise ValueError(f"Could not read image file: {first_image_path}")
    image_height_pixels = image.shape[0]
    image_width_pixels = image.shape[1]

    if image_height_pixels <= 0 or image_width_pixels <= 0:
        raise ValueError("Image dimensions must be greater than zero.")

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

    if not all([recording_start_date, recording_start_time]):
        raise ValueError("Could not determine recording start date or time.")
    
    return {
        "total_replicates": total_replicates,
        "recording_start_time": recording_start_time,
        "recording_start_date": recording_start_date,
        "raw_img_paths": raw_img_paths,
        "image_height_pixels": image_height_pixels,
        "image_width_pixels": image_width_pixels,
    }