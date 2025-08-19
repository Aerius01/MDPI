#!/usr/bin/env python3
"""
Command line interface for duplicate detection module.
Usage: python -m modules.duplicate_detection [options]
"""

import argparse
import sys
from dataclasses import dataclass
from modules.common.cli_utils import CommonCLI
from modules.common.constants import CONSTANTS
from .detector import process_images

# Destructured CONSTANTS for cleaner readability
DUPLICATE_DETECTION_DISPLAY_SIZE = CONSTANTS.DUPLICATE_DETECTION_DISPLAY_SIZE
DUPLICATE_DETECTION_REMOVE = CONSTANTS.DUPLICATE_DETECTION_REMOVE
DUPLICATE_DETECTION_SHOW_MONTAGES = CONSTANTS.DUPLICATE_DETECTION_SHOW_MONTAGES

@dataclass
class ValidatedArguments:
    image_paths: list
    display_size: tuple
    remove_flag: bool
    show_montages: bool

def process_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Handle montage display logic
    show_montages = args.show_montages and not args.no_montages
    
    # Get image paths from input directory
    print(f"[DUPLICATES]: Loading images from {args.input}")
    image_paths = CommonCLI.get_image_group_from_folder(args.input)
    print(f"[DUPLICATES]: Found {len(image_paths)} images")
    
    return ValidatedArguments(
        image_paths=image_paths,
        display_size=args.display_size,
        remove_flag=args.remove,
        show_montages=show_montages
    )

def main(validated_arguments: ValidatedArguments):
    # Run duplicate detection
    removed_paths = process_images(validated_arguments.image_paths, validated_arguments.display_size, validated_arguments.remove_flag, validated_arguments.show_montages)
    removed_count = len(removed_paths)
    
    print(f"[DUPLICATES]: Processing completed.")
    if validated_arguments.remove_flag:
        print(f"[DUPLICATES]: {removed_count} duplicate images were removed.")
    else:
        print(f"[DUPLICATES]: {removed_count} duplicate images were found (but not removed).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect and optionally remove duplicate images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.duplicate_detection -i ./images
  python -m modules.duplicate_detection -i ./images --remove
  python -m modules.duplicate_detection -i ./images --remove --no-montages
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images')
    parser.add_argument('--remove', action='store_true',
                        help='Remove duplicate images (keep only first occurrence)')
    parser.add_argument('--show-montages', action='store_true', default=DUPLICATE_DETECTION_SHOW_MONTAGES,
                        help='Show montages of duplicate images (default: True)')
    parser.add_argument('--no-montages', action='store_true',
                        help='Do not show montages of duplicate images')
    parser.add_argument('--display-size', nargs=2, type=int, default=DUPLICATE_DETECTION_DISPLAY_SIZE,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Display size for montages (default: 500 500)')
    
    args = parser.parse_args()
    
    try:
        validated_arguments = process_arguments(args)
        main(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)