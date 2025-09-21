#!/usr/bin/env python3
"""
Unified CLI to run the full MDPI processing pipeline end-to-end.

Steps:
1) Depth profiling
2) Flatfielding
3) Object detection
4) Object classification
5) Concentration calculation & plotting

Example:
  python3 run_pipeline.py \
    -i /home/david-james/Desktop/04-MDPI/MDPI/profiles/Project_Example/20230425/day/E01_01 \
    -m /home/david-james/Desktop/04-MDPI/MDPI/model
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Common
from modules.common.cli_utils import prompt_for_mdpi_configuration
from modules.common.parser import parse_file_metadata, find_single_csv_file


# Modules
from modules.depth_profiling.depth_profile_data import (
    CAPTURE_RATE,
    IMAGE_HEIGHT_CM,
    read_csv_with_encodings,
    detect_camera_format,
    PRESSURE_SENSOR_CSV_SEPARATOR,
)
from modules.depth_profiling.run import run_depth_profiling
from modules.flatfielding.run import run_flatfielding
from modules.object_detection.run import run_detection
from modules.object_classification.run import run_classification
from modules.plotter.run import run_plotter

# Global configuration defaults (not in CONSTANTS)
# Volume in L == dm^3
DEFAULT_IMG_DEPTH = 1.00 # in decimeters
DEFAULT_IMG_WIDTH = IMAGE_HEIGHT_CM / 10.0 # in decimeters, equal to IMAGE_HEIGHT_CM
MODEL_CHECKPOINT_FILENAME = 'model.ckpt' # The filename of the model checkpoint file
MODEL_CHECKPOINT_EXTENSIONS = ['meta', 'index', 'data-00000-of-00001'] # The extensions of the required model checkpoint files


def _validate_model(model_path: str):
    # Ensure the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    
    # Ensure the required model checkpoint files exist
    model_checkpoint = os.path.join(model_path, MODEL_CHECKPOINT_FILENAME)
    if not any(os.path.exists(f"{model_checkpoint}.{ext}") for ext in MODEL_CHECKPOINT_EXTENSIONS):
        raise FileNotFoundError(f"Model checkpoint files not found in: {model_path}")


def validate_inputs_and_setup(input_dir, model_dir, capture_rate, image_height_cm, img_depth, img_width):
    """
    Validates all pipeline inputs and sets up necessary configurations.
    """
    # Resolve absolute paths
    input_dir_path = Path(input_dir).resolve()
    model_dir_path = Path(model_dir).resolve()

    # Set and validate output directory
    output_root = os.path.join(input_dir_path, 'output')
    output_root_path = Path(output_root).resolve()

    # Create and validate output directory
    try:
        os.makedirs(output_root_path, exist_ok=True)
        if not os.access(output_root_path, os.W_OK):
            raise PermissionError(f"Output directory '{output_root_path}' is not writable.")
    except Exception as e:
        raise OSError(f"Error creating or validating output directory '{output_root_path}': {e}") from e

    # Centralized metadata parsing
    print("[PIPELINE]: Parsing metadata from input directory...")
    metadata = parse_file_metadata(input_dir_path)

    # Find the pressure sensor CSV file
    pressure_sensor_csv_path = find_single_csv_file(input_dir_path)
    header_df = read_csv_with_encodings(
        pressure_sensor_csv_path,
        sep=PRESSURE_SENSOR_CSV_SEPARATOR,
        header=0,
        engine='python',
        nrows=0
    )
    camera_format = detect_camera_format(header_df)
    print(f"[PIPELINE]: Detected {camera_format} camera format.")

    # Validate model directory
    _validate_model(str(model_dir_path))

    # Create and return run_config with resolved paths and new data
    return SimpleNamespace(
        input_dir=str(input_dir_path),
        model_dir=str(model_dir_path),
        output_root=str(output_root_path),
        capture_rate=capture_rate,
        image_height_cm=image_height_cm,
        img_depth=img_depth,
        img_width=img_width,
        metadata=metadata,
        pressure_sensor_csv_path=pressure_sensor_csv_path,
        camera_format=camera_format,
    )


def execute_pipeline(run_config: SimpleNamespace):
    """
    Executes the full MDPI pipeline.
    This function contains the core logic and is called by both the CLI and the Streamlit app.
    """
    # 1) Depth profiling
    print("[PIPELINE]: Running depth profiling...")
    depth_df = run_depth_profiling(run_config)

    # 2) Flatfielding
    print("[PIPELINE]: Running flatfielding...")
    run_flatfielding(run_config, depth_df)

    # 3) Object detection
    print("[PIPELINE]: Running object detection...")
    object_data_df = run_detection(run_config, depth_df)

    # 4) Classification
    print("[PIPELINE]: Running object classification...")
    object_data_df = run_classification(
        run_config=run_config,
        object_data_df=object_data_df,
    )

    # 5) Concentration calculation & plotting
    print("[PIPELINE]: Running plotter...")
    run_plotter(
        run_config=run_config,
        object_data_df=object_data_df,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the full MDPI pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py -i ./profiles/Project_Example/20230425/day/E01_01 -m ./model
        """,
    )

    # Required/primary args
    parser.add_argument("-i", "-\u002Dinput", dest="input", required=True, help="Input directory with raw MDPI images and pressure sensor CSV")
    parser.add_argument("-m", "-\u002Dmodel", dest="model", required=True, help="Path to trained model directory containing model.ckpt files")

    args = parser.parse_args()

    # Pass defaults in centimeters to the prompt; it will convert back to dm
    capture_rate, image_height_cm, img_depth, img_width = prompt_for_mdpi_configuration(
        CAPTURE_RATE, IMAGE_HEIGHT_CM, DEFAULT_IMG_DEPTH * 10.0, DEFAULT_IMG_WIDTH * 10.0
    )

    try:
        run_config = validate_inputs_and_setup(
            input_dir=args.input,
            model_dir=args.model,
            capture_rate=capture_rate,
            image_height_cm=image_height_cm,
            img_depth=img_depth,
            img_width=img_width,
        )
        execute_pipeline(run_config)
        print("[PIPELINE]: All steps completed successfully!")
    except Exception as e:
        print(f"[PIPELINE]: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


