#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python3 -m modules.depth_profiling [options]
"""
import argparse
import sys
from pathlib import Path
import os
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

def main():
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

    # Validate capture rate
    if args.capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")

    # Get image paths from input directory
    input_path = Path(args.input)
    metadata = parse_metadata(input_path)
    pressure_sensor_csv_path = find_single_csv_file(input_path)

    print(f"[PROFILING]: Found {len(metadata['raw_img_paths'])} images")

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"])
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Some additional metadata processing
        start_datetime = datetime.datetime.combine(metadata['recording_start_date'], metadata['recording_start_time'])
        
        # Run depth profiling
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
        df = profiler.map_images_to_depths(metadata['raw_img_paths'], pressure_sensor_csv_path, start_datetime, args.capture_rate)
        
        # Save depth data to CSV
        if df is not None:
            output_csv_path = os.path.join(output_path, "depth_profiles" + CSV_EXTENSION)
            df.to_csv(output_csv_path, index=False)
            print(f"[PROFILING]: Successfully saved data to {output_csv_path}")
            print(f"[PROFILING]: Processing completed successfully!")
        else:
            raise Exception("Failed to process depth data.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 