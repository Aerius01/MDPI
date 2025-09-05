#!/usr/bin/env python3
"""
Command line interface for duplicate detection module.
Usage: python -m modules.duplicate_detection [options]
"""

import argparse
import sys
from typing import List
from .utils import process_arguments
from .detector import deduplicate_images

def main(image_paths: List[str]):
    # Run duplicate detection
    removed_paths = deduplicate_images(image_paths)
    removed_count = len(removed_paths)
    
    print(f"[DUPLICATES]: Processing completed.")
    print(f"[DUPLICATES]: {removed_count} duplicate images were removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect and remove duplicate images from a folder.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.duplicate_detection -i ./images
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images to deduplicate')
    
    args = parser.parse_args()
    
    try:
        validated_arguments = process_arguments(args)
        main(validated_arguments)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)