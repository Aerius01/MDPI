#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python3 -m modules.depth_profiling [options]
"""
import argparse
import sys

from .depth_profile_data import process_arguments
from .profiler import profile_depths

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

    try:
        # All parameters are now automatically detected and configured
        validated_arguments = process_arguments(args)
        profile_depths(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 