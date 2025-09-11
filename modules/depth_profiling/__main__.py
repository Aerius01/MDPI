#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python3 -m modules.depth_profiling [options]
"""
import argparse
import sys

from .depth_profile_data import process_arguments
from .profiler import profile_depths
from .depth_profile_data import CAPTURE_RATE, IMAGE_HEIGHT_CM
from ..common.cli_utils import prompt_for_mdpi_configuration

def main():
    """Main function to run the depth profiling process."""
    capture_rate, image_height_cm = prompt_for_mdpi_configuration(CAPTURE_RATE, IMAGE_HEIGHT_CM)

    parser = argparse.ArgumentParser(
        description='Process images for depth profiling using CSV depth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m modules.depth_profiling -i ./raw_images_folder
  python3 -m modules.depth_profiling -i ./raw_images_folder -o ./output
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help="Input directory containing the raw MDPI images (any of: '.png', '.jpg', '.jpeg', or '.tiff' format) and the pressure sensor (depth) data .csv file")
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the image-depth mapping csv file. The full path will be <output_directory>/project/date/cycle/location/depth_profiles.csv')
    
    args = parser.parse_args()

    try:
        # All parameters are now automatically detected and configured
        validated_arguments = process_arguments(
            args,
            capture_rate_override=capture_rate,
            image_height_cm_override=image_height_cm
        )
        profile_depths(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 