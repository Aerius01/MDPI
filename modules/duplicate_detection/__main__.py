#!/usr/bin/env python3
"""
Command line interface for duplicate detection module.
Usage: python -m modules.duplicate_detection [options]
"""

import argparse
import sys
from modules.common.constants import CONSTANTS
from .deduplication_data import DeduplicationData, process_arguments
from .detector import deduplicate_images

# Destructured CONSTANTS for cleaner readability
DUPLICATE_DETECTION_DISPLAY_SIZE = CONSTANTS.DUPLICATE_DETECTION_DISPLAY_SIZE
DUPLICATE_DETECTION_REMOVE = CONSTANTS.DUPLICATE_DETECTION_REMOVE
DUPLICATE_DETECTION_SHOW_MONTAGES = CONSTANTS.DUPLICATE_DETECTION_SHOW_MONTAGES

def main(validated_arguments: DeduplicationData):
    # Run duplicate detection
    removed_paths = deduplicate_images(validated_arguments)
    removed_count = len(removed_paths)
    
    print(f"[DUPLICATES]: Processing completed.")
    if validated_arguments.remove:
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