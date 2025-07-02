#!/usr/bin/env python3
"""
Command line interface for duplicate detection module.
Usage: python -m modules.duplicate_detection [options]
"""

import argparse
import sys
from modules.common.cli_utils import CommonCLI
from modules.common.constants import get_image_sort_key
from .config import DuplicateConfig
from .detector import DuplicateDetector

def main():
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
    parser.add_argument('--show-montages', action='store_true', default=True,
                        help='Show montages of duplicate images (default: True)')
    parser.add_argument('--no-montages', action='store_true',
                        help='Do not show montages of duplicate images')
    parser.add_argument('--display-size', nargs=2, type=int, default=[500, 500],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Display size for montages (default: 500 500)')
    
    args = parser.parse_args()
    
    # Handle montage display logic
    show_montages = args.show_montages and not args.no_montages
    
    try:
        # Get image paths from input directory
        print(f"[DUPLICATES]: Loading images from {args.input}")
        image_paths = CommonCLI.get_image_group_from_folder(args.input, get_image_sort_key)
        print(f"[DUPLICATES]: Found {len(image_paths)} images")
        
        # Configure duplicate detector
        config = DuplicateConfig(
            remove=args.remove,
            display_size=tuple(args.display_size),
            show_montages=show_montages
        )
        
        # Run duplicate detection
        detector = DuplicateDetector(config)
        removed_count = detector.process_group(image_paths)
        
        print(f"[DUPLICATES]: Processing completed.")
        if args.remove:
            print(f"[DUPLICATES]: {removed_count} duplicate images were removed.")
        else:
            print(f"[DUPLICATES]: {removed_count} duplicate images were found (but not removed).")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 