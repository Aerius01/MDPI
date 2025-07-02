#!/usr/bin/env python3
"""
Command line interface for object detection module.
Usage: python -m modules.object_detection [options]
"""

import argparse
import os
import sys
from modules.common.cli_utils import CommonCLI
from modules.common.constants import get_image_sort_key
from .config import DetectionConfig
from .detector import ObjectDetector

def main():
    parser = argparse.ArgumentParser(
        description='Detect objects in images and extract vignettes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_detection -i ./flatfielded
  python -m modules.object_detection -i ./flatfielded -o ./output/vignettes --threshold-value 180
  python -m modules.object_detection -i ./flatfielded -o ./output/vignettes --min-object-size 100 --max-object-size 3000
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images for object detection')
    parser.add_argument('-o', '--output', default='./output/vignettes',
                        help='Top-level output directory for detected object vignettes')
    parser.add_argument('--threshold-value', type=int, default=190,
                        help='Binary threshold value (default: 190)')
    parser.add_argument('--threshold-max', type=int, default=255,
                        help='Maximum threshold value (default: 255)')
    parser.add_argument('--min-object-size', type=int, default=75,
                        help='Minimum object size in pixels (default: 75)')
    parser.add_argument('--max-object-size', type=int, default=5000,
                        help='Maximum object size in pixels (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing images (default: 10)')
    parser.add_argument('--max-eccentricity', type=float, default=0.97,
                        help='Maximum eccentricity for region filtering (default: 0.97)')
    parser.add_argument('--max-mean-intensity', type=int, default=130,
                        help='Maximum mean intensity for region filtering (default: 130)')
    parser.add_argument('--min-major-axis-length', type=int, default=25,
                        help='Minimum major axis length for region filtering (default: 25)')
    parser.add_argument('--max-min-intensity', type=int, default=65,
                        help='Maximum minimum intensity for region filtering (default: 65)')
    
    args = parser.parse_args()
    
    try:
        # Validate and create output path
        output_path = CommonCLI.validate_output_path(args.output)
        
        # Get image paths from input directory
        print(f"[DETECTION]: Loading images from {args.input}")
        image_paths = CommonCLI.get_image_group_from_folder(args.input, get_image_sort_key)
        print(f"[DETECTION]: Found {len(image_paths)} images")
        
        # Configure object detector
        config = DetectionConfig(
            threshold_value=args.threshold_value,
            threshold_max=args.threshold_max,
            min_object_size=args.min_object_size,
            max_object_size=args.max_object_size,
            batch_size=args.batch_size,
            max_eccentricity=args.max_eccentricity,
            max_mean_intensity=args.max_mean_intensity,
            min_major_axis_length=args.min_major_axis_length,
            max_min_intensity=args.max_min_intensity
        )
        
        # Run object detection
        detector = ObjectDetector(config)
        output_files = detector.process_group(image_paths, output_path)
        
        print(f"[DETECTION]: Processing completed successfully!")
        if output_files:
            # Extract the directory path from the first output file
            output_dir = os.path.dirname(output_files[0])
            print(f"[DETECTION]: {len(output_files)} vignettes saved to {output_dir}")
        else:
            print(f"[DETECTION]: No files were processed")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 