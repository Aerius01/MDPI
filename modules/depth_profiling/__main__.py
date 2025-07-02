#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python -m modules.depth_profiling [options]
"""

import argparse
import sys
from modules.common.cli_utils import CommonCLI
from modules.common.constants import get_image_sort_key
from .config import ProfileConfig
from .profiler import DepthProfiler

def main():
    parser = argparse.ArgumentParser(
        description='Process images for depth profiling using CSV depth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.depth_profiling -i ./raw_images
  python -m modules.depth_profiling -i ./raw_images -o ./output/depth_profiles --depth-multiplier 5.0
  python -m modules.depth_profiling -i ./raw_images -o ./output/depth_profiles --csv-separator ","
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images and CSV depth data')
    parser.add_argument('-o', '--output', default='./output/depth_profiles',
                        help='Top-level output directory for depth-profiled images')
    parser.add_argument('--csv-separator', default=';',
                        help='CSV separator character (default: ;)')
    parser.add_argument('--csv-header-row', type=int, default=6,
                        help='CSV header row index (default: 6)')
    parser.add_argument('--csv-skipfooter', type=int, default=1,
                        help='Number of CSV footer rows to skip (default: 1)')
    parser.add_argument('--depth-multiplier', type=float, default=10.0,
                        help='Depth multiplier for CSV values (default: 10.0)')
    parser.add_argument('--csv-columns', nargs=2, type=int, default=[0, 1],
                        metavar=('TIME_COL', 'DEPTH_COL'),
                        help='CSV column indices for time and depth (default: 0 1)')
    
    args = parser.parse_args()
    
    try:
        # Validate and create output path
        output_path = CommonCLI.validate_output_path(args.output)
        
        # Get image paths from input directory
        print(f"[PROFILING]: Loading images from {args.input}")
        image_paths = CommonCLI.get_image_group_from_folder(args.input, get_image_sort_key)
        print(f"[PROFILING]: Found {len(image_paths)} images")
        
        # Configure depth profiler
        config = ProfileConfig(
            csv_separator=args.csv_separator,
            csv_header_row=args.csv_header_row,
            csv_columns=args.csv_columns,
            csv_skipfooter=args.csv_skipfooter,
            depth_multiplier=args.depth_multiplier
        )
        
        # Run depth profiling
        profiler = DepthProfiler(config)
        output_files = profiler.process_group(image_paths, output_path)
        
        print(f"[PROFILING]: Processing completed successfully!")
        print(f"[PROFILING]: {len(output_files)} files saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 