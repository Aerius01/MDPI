#!/usr/bin/env python3
"""
Command line interface for object classification module.
Usage: python -m modules.object_classification [options]
"""

import argparse
import os
import sys
from modules.common.cli_utils import CommonCLI
from .classifier import classify_objects

def main():
    parser = argparse.ArgumentParser(
        description='Classify objects in vignette images using a trained CNN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_classification -i ./vignettes -m ./model
  python -m modules.object_classification -i ./vignettes -m ./model --batch-size 64
  python -m modules.object_classification -i ./vignettes -m ./model -o ./output/classification --input-size 64
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing vignette images to classify')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to trained model directory containing model.ckpt')
    parser.add_argument('-o', '--output', default='./output/classification',
                        help='Top-level output directory for classification results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for classification (default: 32)')
    parser.add_argument('--input-size', type=int, default=50,
                        help='Input image size for the model (default: 50)')
    parser.add_argument('--input-depth', type=int, default=1,
                        help='Input image depth (channels) for the model (default: 1)')
    
    args = parser.parse_args()
    
    try:
        # Validate paths
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model directory does not exist: {args.model}")
        
        model_checkpoint = os.path.join(args.model, "model.ckpt")
        if not any(os.path.exists(f"{model_checkpoint}.{ext}") for ext in ['meta', 'index', 'data-00000-of-00001']):
            raise FileNotFoundError(f"Model checkpoint files not found in: {args.model}")
        
        # Validate and create output path
        output_path = CommonCLI.validate_output_path(args.output)
        
        # Get image paths from input directory
        print(f"[CLASSIFICATION]: Loading images from {args.input}")
        image_paths = CommonCLI.get_image_group_from_folder(args.input)
        print(f"[CLASSIFICATION]: Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("[CLASSIFICATION]: No images found to classify.")
            return
        
        # Run classification
        classify_objects(
            image_group=image_paths,
            output_path=output_path,
            model_path=args.model,
            batch_size=args.batch_size,
            input_size=args.input_size,
            input_depth=args.input_depth
        )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 