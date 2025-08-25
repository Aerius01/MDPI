#!/usr/bin/env python3
"""
Unified CLI to run the full MDPI processing pipeline end-to-end.

Steps:
1) Depth profiling
2) Flatfielding
3) Object detection
4) Object classification
5) Concentration calculation

Example:
  python3 run_pipeline.py \
    -i /home/david-james/Desktop/04-MDPI/MDPI/profiles/Project_Example/20230425/day/E01_01 \
    -o /home/david-james/Desktop/04-MDPI/MDPI/output \
    -c 2.4 \
    -m /home/david-james/Desktop/04-MDPI/MDPI/model
"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Common
from modules.common.constants import CONSTANTS

# Depth profiling
from modules.depth_profiling.depth_profile_data import (
    CsvParams,
    DepthParams,
    process_arguments as depth_process_arguments,
)
from modules.depth_profiling.profiler import profile_depths

# Flatfielding
from modules.flatfielding.flatfielding_data import (
    process_arguments as flatfielding_process_arguments,
)
from modules.flatfielding.flatfielding import flatfield_images

# Object detection
from modules.object_detection.detection_data import (
    validate_arguments as detection_validate_arguments,
)
from modules.object_detection.__main__ import run_detection
from modules.object_detection.detector import Detector
from modules.object_detection.output_handler import OutputHandler

# Classification
from modules.object_classification.utils import parse_vignette_metadata
from modules.object_classification.classification_data import validate_arguments as cls_validate_arguments
from modules.object_classification.inference_engine import InferenceEngine
from modules.object_classification.processor import ClassificationProcessor
from modules.object_classification.run import run_classification

# Plotter
from modules.plotter.calculate_concentrations import (
    ConcentrationConfig,
    calculate_concentration_data,
)
import pandas as pd

# Global configuration defaults (not in CONSTANTS)
DEFAULT_BIN_SIZE = 0.1
DEFAULT_MAX_DEPTH = 18.0
DEFAULT_CONCENTRATION_GROUPS = "copepod,cladocera"
DEFAULT_IMG_DEPTH = 10.0
DEFAULT_IMG_WIDTH = 0.42
TIME_COLUMN_NAME = "time"
DEPTH_COLUMN_NAME = "depth"
CONCENTRATION_OUTPUT_FILENAME = "concentration_data.csv"
OBJECT_DATA_CSV_FILENAME = "object_data.csv"


def run_depth_profiling(input_dir: str, output_root: str, capture_rate: float) -> str:
    """Run depth profiling and return the base output path for this run."""
    # Build CSV and depth params from constants
    csv_params = CsvParams(
        separator=CONSTANTS.CSV_SEPARATOR,
        header_row=CONSTANTS.CSV_HEADER_ROW,
        columns=CONSTANTS.CSV_COLUMNS,
        skipfooter=CONSTANTS.CSV_SKIPFOOTER,
        extension=CONSTANTS.CSV_EXTENSION,
        time_column_name=TIME_COLUMN_NAME,
        depth_column_name=DEPTH_COLUMN_NAME,
    )
    depth_params = DepthParams(
        pressure_sensor_depth_multiplier=CONSTANTS.PRESSURE_SENSOR_DEPTH_MULTIPLIER,
        image_height_cm=CONSTANTS.IMAGE_HEIGHT_CM,
        image_height_pixels=CONSTANTS.IMAGE_HEIGHT_PIXELS,
        overlap_correction_depth_multiplier=CONSTANTS.OVERLAP_CORRECTION_DEPTH_MULTIPLIER,
    )

    args = SimpleNamespace(input=input_dir, output=output_root, capture_rate=capture_rate)
    validated = depth_process_arguments(args, csv_params, depth_params)

    profile_depths(validated)

    # Depth module writes to <output_root>/<project>/<date>/<cycle>/<location>
    return validated.output_path


def run_flatfielding(raw_input_dir: str, depth_csv_path: str, output_root: str) -> str:
    """Run flatfielding and return the flatfielded images directory path."""
    args = SimpleNamespace(input=raw_input_dir, depth_profiles=depth_csv_path, output=output_root)
    data = flatfielding_process_arguments(args)
    flatfield_images(data)
    return data.output_path


def run_detection_step(flatfield_dir: str, depth_csv_path: str, output_root: str) -> str:
    """Run detection and return the vignettes output directory."""
    args = SimpleNamespace(input=flatfield_dir, depth_profiles=depth_csv_path, output=output_root)
    detection_data = detection_validate_arguments(args)

    detector = Detector(
        threshold_value=CONSTANTS.THRESHOLD_VALUE,
        threshold_max=CONSTANTS.THRESHOLD_MAX,
        min_object_size=CONSTANTS.MIN_OBJECT_SIZE,
        max_object_size=CONSTANTS.MAX_OBJECT_SIZE,
        max_eccentricity=CONSTANTS.MAX_ECCENTRICITY,
        max_mean_intensity=CONSTANTS.MAX_MEAN_INTENSITY,
        min_major_axis_length=CONSTANTS.MIN_MAJOR_AXIS_LENGTH,
        max_min_intensity=CONSTANTS.MAX_MIN_INTENSITY,
        small_object_threshold=CONSTANTS.SMALL_OBJECT_THRESHOLD,
        medium_object_threshold=CONSTANTS.MEDIUM_OBJECT_THRESHOLD,
        large_object_padding=CONSTANTS.LARGE_OBJECT_PADDING,
        small_object_padding=CONSTANTS.SMALL_OBJECT_PADDING,
        medium_object_padding=CONSTANTS.MEDIUM_OBJECT_PADDING,
        batch_size=CONSTANTS.BATCH_SIZE,
    )
    output_handler = OutputHandler(csv_extension=CONSTANTS.CSV_EXTENSION, csv_separator=CONSTANTS.CSV_SEPARATOR)

    run_detection(detection_data, detector, output_handler)
    return detection_data.output_path


def run_classification_step(
    vignettes_dir: str,
    output_root: str,
    model_dir: str,
    batch_size: int,
    input_size: int,
    input_depth: int,
) -> str:
    """Run classification and return the base output directory for files (without 'vignettes')."""
    metadata = parse_vignette_metadata(Path(vignettes_dir))
    args = dict(
        input=vignettes_dir,
        output=output_root,
        model=model_dir,
        batch_size=batch_size,
        input_size=input_size,
        input_depth=input_depth,
    )
    classification_data = cls_validate_arguments(**args, **metadata)

    inference_engine = InferenceEngine(classification_data)
    processor = ClassificationProcessor()
    run_classification(classification_data, inference_engine, processor)

    # Classification writes outputs into <output_root>/<project>/<date>/<cycle>/<location>
    return str(classification_data.output_path)


def run_concentration_step(
    object_data_csv: str,
    max_depth: float,
    bin_size: float,
    groups: list[str],
    img_depth: float,
    img_width: float,
) -> str:
    """Run concentration calculation and return path to saved CSV."""
    # Load classification CSV (merged with detection)
    data = pd.read_csv(object_data_csv, dtype={
        'project': str,
        'cycle': str,
        'replicate': str,
        'prediction': str,
        'label': str,
        'FileName': str,
    })

    config = ConcentrationConfig(
        max_depth=max_depth,
        bin_size=bin_size,
        output_file_name=CONCENTRATION_OUTPUT_FILENAME,
        groups=groups,
        img_depth=img_depth,
        img_width=img_width,
    )
    concentration_df = calculate_concentration_data(data, config)
    output_path = os.path.join(os.path.dirname(object_data_csv), config.output_file_name)
    concentration_df.to_csv(output_path, index=False)
    print(f"[PLOTTER]: Concentration data saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Run the full MDPI pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_pipeline.py -i ./profiles/Project_Example/20230425/day/E01_01 -o ./output -c 2.4 -m ./model
        """,
    )

    # Required/primary args
    parser.add_argument("-i", "-\u002Dinput", dest="input", required=True, help="Input directory with raw MDPI images and pressure sensor CSV")
    parser.add_argument("-o", "-\u002Doutput", dest="output", default="./output", help="Root output directory")
    parser.add_argument("-c", "-\u002Dcapture-rate", dest="capture_rate", type=float, required=True, help="MDPI image capture rate in Hz")
    parser.add_argument("-m", "-\u002Dmodel", dest="model", required=True, help="Path to trained model directory containing model.ckpt files")

    # Optional: classification knobs
    parser.add_argument("--classification-batch-size", type=int, default=CONSTANTS.CLASSIFICATION_BATCH_SIZE)
    parser.add_argument("--classification-input-size", type=int, default=CONSTANTS.CLASSIFICATION_INPUT_SIZE)
    parser.add_argument("--classification-input-depth", type=int, default=CONSTANTS.CLASSIFICATION_INPUT_DEPTH)

    # Optional: concentration knobs (defaults mirror calculate_concentrations module)
    parser.add_argument("--bin-size", type=float, default=DEFAULT_BIN_SIZE)
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--groups", type=str, default=DEFAULT_CONCENTRATION_GROUPS, help="Comma-separated class labels to include")
    parser.add_argument("--img-depth", type=float, default=DEFAULT_IMG_DEPTH)
    parser.add_argument("--img-width", type=float, default=DEFAULT_IMG_WIDTH)

    args = parser.parse_args()

    try:
        # Resolve absolute paths
        input_dir = str(Path(args.input).resolve())
        output_root = str(Path(args.output).resolve())
        model_dir = str(Path(args.model).resolve())

        # 1) Depth profiling
        print("[PIPELINE]: Running depth profiling...")
        base_output_dir = run_depth_profiling(input_dir, output_root, args.capture_rate)
        depth_csv = os.path.join(base_output_dir, f"depth_profiles{CONSTANTS.CSV_EXTENSION}")

        # 2) Flatfielding
        print("[PIPELINE]: Running flatfielding...")
        flatfield_dir = run_flatfielding(input_dir, depth_csv, output_root)

        # 3) Object detection
        print("[PIPELINE]: Running object detection...")
        vignettes_dir = run_detection_step(flatfield_dir, depth_csv, output_root)

        # 4) Classification
        print("[PIPELINE]: Running object classification...")
        classification_output_dir = run_classification_step(
            vignettes_dir=vignettes_dir,
            output_root=output_root,
            model_dir=model_dir,
            batch_size=args.classification_batch_size,
            input_size=args.classification_input_size,
            input_depth=args.classification_input_depth,
        )

        # 5) Concentration calculation
        print("[PIPELINE]: Calculating concentrations...")
        object_data_csv = os.path.join(classification_output_dir, OBJECT_DATA_CSV_FILENAME)
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        run_concentration_step(
            object_data_csv=object_data_csv,
            max_depth=args.max_depth,
            bin_size=args.bin_size,
            groups=groups,
            img_depth=args.img_depth,
            img_width=args.img_width,
        )

        print("[PIPELINE]: All steps completed successfully!")
    except Exception as e:
        print(f"[PIPELINE]: Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


