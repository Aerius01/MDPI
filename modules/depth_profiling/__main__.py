#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python3 -m modules.depth_profiling [options]
"""
import argparse
import sys

from modules.common.constants import CONSTANTS
from .depth_profile_data import CsvParams, DepthParams, process_arguments
from .profiler import profile_depths

# Destructured CONSTANTS for dependency injection
CSV_SEPARATOR = CONSTANTS.CSV_SEPARATOR
CSV_HEADER_ROW = CONSTANTS.CSV_HEADER_ROW
CSV_COLUMNS = CONSTANTS.CSV_COLUMNS
CSV_SKIPFOOTER = CONSTANTS.CSV_SKIPFOOTER
CSV_EXTENSION = CONSTANTS.CSV_EXTENSION
PRESSURE_SENSOR_DEPTH_MULTIPLIER = CONSTANTS.PRESSURE_SENSOR_DEPTH_MULTIPLIER
IMAGE_HEIGHT_CM = CONSTANTS.IMAGE_HEIGHT_CM
IMAGE_HEIGHT_PIXELS = CONSTANTS.IMAGE_HEIGHT_PIXELS
OVERLAP_CORRECTION_DEPTH_MULTIPLIER = CONSTANTS.OVERLAP_CORRECTION_DEPTH_MULTIPLIER
TIME_COLUMN_NAME = "time"
DEPTH_COLUMN_NAME = "depth"

def main():
    """Main function to run the depth profiling process."""
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

    csv_params = CsvParams(
        separator=CSV_SEPARATOR,
        header_row=CSV_HEADER_ROW,
        columns=CSV_COLUMNS,
        skipfooter=CSV_SKIPFOOTER,
        extension=CSV_EXTENSION,
        time_column_name=TIME_COLUMN_NAME,
        depth_column_name=DEPTH_COLUMN_NAME
    )
    
    depth_params = DepthParams(
        pressure_sensor_depth_multiplier=PRESSURE_SENSOR_DEPTH_MULTIPLIER,
        image_height_cm=IMAGE_HEIGHT_CM,
        image_height_pixels=IMAGE_HEIGHT_PIXELS,
        overlap_correction_depth_multiplier=OVERLAP_CORRECTION_DEPTH_MULTIPLIER
    )

    try:
        validated_arguments = process_arguments(
            args, 
            csv_params, 
            depth_params
        )
        profile_depths(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 