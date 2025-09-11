#!/usr/bin/env python3
"""
Command line interface for flatfielding module.
Usage: python3 -m modules.flatfielding [options]
"""

import argparse
import sys
from .flatfielding import flatfield_images
from .flatfielding_data import process_arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply flatfielding correction to images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m modules.flatfielding -i ./raw_images_folder
  python3 -m modules.flatfielding -i ./raw_images_folder -o ./output
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help="Input directory containing the raw MDPI images (any of: '.png', '.jpg', '.jpeg', or '.tiff' format)")
    parser.add_argument('-d', '--depth-profiles', required=True,
                        help='Path to the depth profiles CSV file which includes the pixel overlap data.')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the flatfielded images. The full path for the images will be <output_directory>/project/date/cycle/location/flatfielded_images')
    
    args = parser.parse_args()
    
    try:
        flatfielding_data = process_arguments(args)
        flatfield_images(flatfielding_data)
        
    except Exception as e:
        raise e
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 
