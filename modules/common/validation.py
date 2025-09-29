import os
from pathlib import Path

from modules.common.parser import find_single_csv_file, parse_file_metadata
from modules.depth_profiling.depth_profile_data import (
    PRESSURE_SENSOR_CSV_SEPARATOR,
    detect_camera_format,
    read_csv_with_encodings,
)

# Constants for model validation
MODEL_CHECKPOINT_FILENAME = "model.ckpt"
MODEL_CHECKPOINT_EXTENSIONS = ["meta", "index", "data-00000-of-00001"]


def validate_input_directory(input_dir: str):
    """
    Validates the structure and content of an MDPI input directory.
    This function checks for the existence of a directory, a single pressure
    sensor CSV file, supported image files, and parses necessary metadata.
    Args:
        input_dir: The path to the input directory.
    Returns:
        A tuple containing:
        - results (list): A list of tuples, where each tuple is a (bool, str)
                          representing a validation check's success and message.
        - metadata (dict): Parsed metadata from file naming conventions.
        - pressure_sensor_csv_path (str): The path to the pressure sensor CSV.
        - camera_format (str): The detected camera format.
    """
    results = []
    metadata = {}
    camera_format = None
    pressure_sensor_csv_path = None
    input_dir_path = Path(input_dir)

    if not input_dir or not os.path.isdir(input_dir_path):
        results.append((False, "Path must be a valid directory."))
        return results, metadata, None, None

    results.append((True, "Path is a valid directory."))

    try:
        metadata = parse_file_metadata(input_dir_path)
        results.append(
            (True, f"Found {metadata.get('total_replicates', 0)} image file(s).")
        )
    except ValueError as e:
        results.append((False, str(e)))
        return results, metadata, None, None

    try:
        pressure_sensor_csv_path = find_single_csv_file(input_dir_path)
        results.append((True, "Found one pressure sensor CSV file."))
    except ValueError as e:
        results.append((False, str(e)))
        return results, metadata, None, None

    # Detect camera format from CSV
    try:
        header_df = read_csv_with_encodings(
            pressure_sensor_csv_path,
            sep=PRESSURE_SENSOR_CSV_SEPARATOR,
            header=0,
            engine="python",
            nrows=0,
        )
        camera_format = detect_camera_format(header_df)
    except Exception as e:
        results.append((False, f"Could not determine camera format from CSV: {e}"))

    return (
        results,
        metadata,
        str(pressure_sensor_csv_path) if pressure_sensor_csv_path else None,
        camera_format,
    )


def validate_model(model_path: str):
    """
    Validates the existence of the model directory and checkpoint files.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    model_checkpoint = os.path.join(model_path, MODEL_CHECKPOINT_FILENAME)
    if not any(
        os.path.exists(f"{model_checkpoint}.{ext}")
        for ext in MODEL_CHECKPOINT_EXTENSIONS
    ):
        raise FileNotFoundError(f"Model checkpoint files not found in: {model_path}")


def setup_output_directory(input_dir: str) -> str:
    """Creates and validates the output directory."""
    input_dir_path = Path(input_dir).resolve()
    output_root = os.path.join(input_dir_path, "output")
    output_root_path = Path(output_root).resolve()

    try:
        os.makedirs(output_root_path, exist_ok=True)
        if not os.access(output_root_path, os.W_OK):
            raise PermissionError(
                f"Output directory '{output_root_path}' is not writable."
            )
    except Exception as e:
        raise OSError(
            f"Error creating or validating output directory '{output_root_path}': {e}"
        ) from e

    return str(output_root_path)
