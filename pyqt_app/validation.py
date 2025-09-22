import os
from pathlib import Path

from modules.common.parser import parse_file_metadata, find_single_csv_file
from modules.depth_profiling.depth_profile_data import (
    PRESSURE_SENSOR_CSV_SEPARATOR,
    detect_camera_format,
    read_csv_with_encodings,
)

def get_detailed_validation_results(path: str):
    """
    Performs a step-by-step validation of an input directory and returns
    a list of results for UI display. Does not raise exceptions.
    
    Returns:
        A tuple containing:
        - results (list): A list of (bool, str) tuples for each validation step.
        - metadata (dict or None): Parsed metadata if successful.
        - camera_format (str or None): Detected camera format if successful.
    """
    results = []
    metadata = None
    camera_format = None
    path_obj = Path(path)

    # 1. Check if it's a directory
    if not path or not os.path.isdir(path_obj):
        results.append((False, "Path must be a valid directory."))
        return results, metadata, camera_format
    results.append((True, "Path is a valid directory."))

    # 2. Check for images and parse metadata
    try:
        metadata = parse_file_metadata(path_obj)
        results.append((True, f"Found {metadata.get('total_replicates', 0)} image file(s)."))
    except ValueError as e:
        results.append((False, str(e)))
        return results, metadata, camera_format
    
    # 3. Check for pressure sensor CSV
    pressure_sensor_csv_path = None
    try:
        pressure_sensor_csv_path = find_single_csv_file(path_obj)
        results.append((True, "Found one pressure sensor CSV file."))
    except (FileNotFoundError, ValueError) as e:
        results.append((False, str(e)))
        return results, metadata, camera_format

    # 4. Detect camera format
    try:
        header_df = read_csv_with_encodings(
            pressure_sensor_csv_path,
            sep=PRESSURE_SENSOR_CSV_SEPARATOR,
            header=0,
            engine="python",
            nrows=0,
        )
        camera_format = detect_camera_format(header_df)
    except Exception:
        # This is not a fatal error for path validation, but good to know
        results.append((False, "Could not determine camera format."))

    return results, metadata, camera_format
