#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python3 -m modules.depth_profiling [options]
"""
import argparse
import sys
from pathlib import Path
import os
from dataclasses import dataclass
from modules.common.cli_utils import CommonCLI
from modules.common.parser import parse_metadata, find_single_csv_file
from .profiler import DepthProfiler
import datetime
from modules.common.constants import CONSTANTS

# Destructured CONSTANTS for cleaner readability
CSV_EXTENSION = CONSTANTS.CSV_EXTENSION
CSV_SEPARATOR = CONSTANTS.CSV_SEPARATOR
CSV_HEADER_ROW = CONSTANTS.CSV_HEADER_ROW
CSV_COLUMNS = CONSTANTS.CSV_COLUMNS
CSV_SKIPFOOTER = CONSTANTS.CSV_SKIPFOOTER
PRESSURE_SENSOR_DEPTH_MULTIPLIER = CONSTANTS.PRESSURE_SENSOR_DEPTH_MULTIPLIER
TIME_COLUMN_NAME = "time"
DEPTH_COLUMN_NAME = "depth"

@dataclass
class ValidatedArguments:
    metadata: dict
    pressure_sensor_csv_path: str
    output_path: str
    capture_rate: float

def process_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Validate capture rate
    if args.capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")

    # Get image paths from input directory
    input_path = Path(args.input)
    metadata = parse_metadata(input_path)
    pressure_sensor_csv_path = find_single_csv_file(input_path)

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"])
    os.makedirs(output_path, exist_ok=True)
    
    return ValidatedArguments(
        metadata=metadata,
        pressure_sensor_csv_path=pressure_sensor_csv_path,
        output_path=output_path,
        capture_rate=args.capture_rate
    )

def main(validated_arguments: ValidatedArguments):
    metadata = validated_arguments.metadata
    pressure_sensor_csv_path = validated_arguments.pressure_sensor_csv_path
    output_path = validated_arguments.output_path
    capture_rate = validated_arguments.capture_rate

    print(f"[PROFILING]: Found {len(metadata['raw_img_paths'])} images")

    # Some additional metadata processing
    start_datetime = datetime.datetime.combine(metadata['recording_start_date'], metadata['recording_start_time'])
    
    # Run depth profiling
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    print(f"[PROFILING]: Processing group: {metadata['project']}/{date_str}/{metadata['cycle']}/{metadata['location']}")
    profiler = DepthProfiler(
        csv_separator=CSV_SEPARATOR,
        csv_header_row=CSV_HEADER_ROW,
        csv_columns=CSV_COLUMNS,
        csv_skipfooter=CSV_SKIPFOOTER,
        depth_multiplier=PRESSURE_SENSOR_DEPTH_MULTIPLIER,
        time_column_name=TIME_COLUMN_NAME,
        depth_column_name=DEPTH_COLUMN_NAME
    )
    df = profiler.map_images_to_depths(metadata['raw_img_paths'], pressure_sensor_csv_path, start_datetime, capture_rate)
    
    # Save depth data to CSV
    if df is not None:
        output_csv_path = os.path.join(output_path, "depth_profiles" + CSV_EXTENSION)
        df.to_csv(output_csv_path, index=False)
        print(f"[PROFILING]: Successfully saved data to {output_csv_path}")
        print(f"[PROFILING]: Processing completed successfully!")
    else:
        raise Exception("Failed to process depth data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Process images for depth profiling using CSV depth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m modules.depth_profiling -i ./raw_images_folder -c 2.4
  python3 -m modules.depth_profiling -i ./raw_images_folder -o ./output -c 2.4
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help="Input directory containing the raw MDPI images (any of: '.png', '.jpg', '.jpeg', or '.tiff' format) and the pressure sensor (depth) data .csv file")
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the image-depth mapping csv file. The full path will be <output_directory>/project/date/cycle/location/depth_profiles.csv')
    parser.add_argument('-c', '--capture-rate', type=float, required=True,
                        help='The image capture rate in hertz (Hz) of the MDPI')
    
    args = parser.parse_args()

    try:
        validated_arguments = process_arguments(args)
        main(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 