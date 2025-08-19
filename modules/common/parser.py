import datetime
import os
from pathlib import Path
from typing import Dict, Any

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
    
    # Get all image files in the directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff'}
    filenames = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in image_extensions]
    if not filenames:
        raise ValueError(f"No image files found in directory '{directory_path}'")

    # Sort by filename
    raw_img_paths = [str(directory_path / filename) for filename in filenames]
    raw_img_paths.sort(key=os.path.basename) 
    
    # Get the last image file in the list
    image_filename = os.path.basename(raw_img_paths[-1])

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

        if len(time_str) != 9:
            raise ValueError(f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm' format.")

        time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
        recording_start_time = datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

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

def _parse_vignette_file_metadata(directory_path: Path) -> Dict[str, Any]:
    """
    Parses metadata from a directory containing vignette images.

    This function is tailored for the object classification module.
    It extracts metadata from the image filenames.

    Args:
        directory_path (Path): The path to the directory containing the vignette images.

    Returns:
        Dict: A dictionary containing the parsed file metadata.
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff'}
    filenames = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in image_extensions]
    if not filenames:
        raise ValueError(f"No image files found in directory '{directory_path}'")

    # Sort filenames to ensure consistent ordering
    filenames.sort()
    
    raw_img_paths = [str(directory_path / filename) for filename in filenames]
    
    # Time is the same for all images, and replicates are parsed from the last one to get the total.
    last_image_filename = filenames[-1]

    try:
        # Parse time and total replicates from the last filename
        # Expected format: HHMMSSmmm_replicate_vignette_objectID.ext
        filename_parts = last_image_filename.split('_')
        time_str = filename_parts[0]
        if len(time_str) != 9:
            raise ValueError(f"Time part of filename '{time_str}' from '{last_image_filename}' has incorrect length. Expected 9 digits for 'HHMMSSmmm' format.")
        
        time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
        recording_start_time = datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

        # Parse total replicates from last filename
        total_replicates = int(filename_parts[1])

    except (IndexError, ValueError) as e:
        raise ValueError(f"Could not parse info from filenames in '{directory_path}'. Expected 'HHMMSSmmm_replicate_vignette_objectID.ext'. Error: {e}")
    
    return {
        "total_replicates": total_replicates,
        "recording_start_time": recording_start_time,
        "raw_img_paths": raw_img_paths
    }

def parse_vignette_metadata(directory_path: Path) -> Dict:
    """Parses directory path and vignette filenames to extract all metadata."""
    path_metadata = _parse_path_metadata(directory_path.parent)
    file_metadata = _parse_vignette_file_metadata(directory_path)
    return {**path_metadata, **file_metadata}

def find_single_csv_file(directory_path: Path) -> str:
    """Finds the single .csv file in the directory and returns its path."""
    p = directory_path
    csv_files = [f for f in os.listdir(p) if f.lower().endswith('.csv')]

    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in directory '{p}'")
    if len(csv_files) > 1:
        raise ValueError(f"Multiple CSV files found in directory '{p}': {', '.join(csv_files)}")

    return str(p / csv_files[0])

def parse_flatfield_metadata_from_directory(directory_path: Path) -> Dict:
    """
    Parses metadata from a directory containing flat-fielded images.

    This function is designed to work with the output of the flatfielding module.
    It extracts metadata from both the directory path and the image filenames.

    Args:
        directory_path (Path): The path to the directory containing the flat-fielded images.
                               The directory should be the 'flatfielded_images' folder.

    Returns:
        Dict: A dictionary containing the parsed metadata.
    """
    path_metadata = _parse_path_metadata(directory_path.parent)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff'}
    filenames = [f for f in os.listdir(directory_path) if os.path.splitext(f)[1].lower() in image_extensions]
    if not filenames:
        raise ValueError(f"No image files found in directory '{directory_path}'")

    # The time is the same for all files in a batch, so we can parse it from the first one.
    time_str = filenames[0].split('_')[0]
    if len(time_str) != 9:
        raise ValueError(f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm' format.")

    time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
    recording_start_time = datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

    # To get total replicates, we parse the number from each filename and find the max.
    total_replicates = 0
    for f in filenames:
        try:
            replicate = int(f.split('_')[1].split('.')[0])
            if replicate > total_replicates:
                total_replicates = replicate
        except (IndexError, ValueError):
            continue  # Ignore files that don't match the format

    raw_img_paths = sorted([str(directory_path / f) for f in filenames])

    file_metadata = {
        "recording_start_time": recording_start_time,
        "total_replicates": total_replicates,
        "raw_img_paths": raw_img_paths,
    }

    return {**path_metadata, **file_metadata} 