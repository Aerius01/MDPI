#!/usr/bin/env python3
"""
Command line interface for flatfielding module.
Usage: python -m modules.flatfielding [options]
"""

import argparse
import os
import sys
from modules.common.cli_utils import CommonCLI
from modules.common.constants import get_image_sort_key
from .config import FlatfieldConfig
from .processor import FlatfieldProcessor

def main():
    parser = argparse.ArgumentParser(
        description='Apply flatfielding correction to images.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.flatfielding -i ./depth_profiles
  python -m modules.flatfielding -i ./depth_profiles -o ./output/flatfielded --batch-size 20
  python -m modules.flatfielding -i ./depth_profiles -o ./output/flatfielded --normalization-factor 200
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images to flatfield')
    parser.add_argument('-o', '--output', default='./output/flatfielded',
                        help='Top-level output directory for flatfielded images')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing images (default: 10)')
    parser.add_argument('--normalization-factor', type=int, default=235,
                        help='Normalization factor for flatfielding (default: 235)')
    parser.add_argument('--output-format', choices=['.jpeg', '.jpg', '.tiff', '.png'], 
                        default='.jpeg',
                        help='Output image format (default: .jpeg)')
    
    args = parser.parse_args()
    
    try:
        # Validate and create output path
        output_path = CommonCLI.validate_output_path(args.output)
        
        # Get image paths from input directory
        print(f"[FLATFIELDING]: Loading images from {args.input}")
        image_paths = CommonCLI.get_image_group_from_folder(args.input, get_image_sort_key)
        print(f"[FLATFIELDING]: Found {len(image_paths)} images")
        
        # Configure flatfield processor
        config = FlatfieldConfig(
            batch_size=args.batch_size,
            normalization_factor=args.normalization_factor,
            output_format=args.output_format
        )
        
        # Run flatfielding
        processor = FlatfieldProcessor(config)
        output_files = processor.process_group(image_paths, output_path)
        
        print(f"[FLATFIELDING]: Processing completed successfully!")
        if output_files:
            # Extract the directory path from the first output file
            output_dir = os.path.dirname(output_files[0])
            print(f"[FLATFIELDING]: {len(output_files)} files saved to {output_dir}")
        else:
            print(f"[FLATFIELDING]: No files were processed")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 