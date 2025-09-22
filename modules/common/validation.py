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
        - metadata (dict): Parsed metadata from file naming conventions.
        - pressure_sensor_csv_path (str): The path to the pressure sensor CSV.
        - camera_format (str): The detected camera format.
    Raises:
        ValueError: If the path is not a valid directory, metadata cannot be
                    parsed, or the camera format cannot be determined.
        FileNotFoundError: If the pressure sensor CSV or image files are missing,
                           or if more than one CSV is found.
    """
    input_dir_path = Path(input_dir)

    if not input_dir or not os.path.isdir(input_dir_path):
        raise ValueError("Path must be a valid directory.")

    metadata = parse_file_metadata(input_dir_path)
    pressure_sensor_csv_path = find_single_csv_file(input_dir_path)

    # Detect camera format from CSV
    header_df = read_csv_with_encodings(
        pressure_sensor_csv_path,
        sep=PRESSURE_SENSOR_CSV_SEPARATOR,
        header=0,
        engine="python",
        nrows=0,
    )
    camera_format = detect_camera_format(header_df)

    return metadata, str(pressure_sensor_csv_path), camera_format


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
