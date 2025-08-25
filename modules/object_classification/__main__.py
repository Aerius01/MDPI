#!/usr/bin/env python3
"""
Command line interface for object classification module.
Usage: python -m modules.object_classification [options]
"""

import argparse
import sys
from .classification_data import validate_arguments
from modules.common.constants import CONSTANTS
from .run import run_classification
from .utils import parse_vignette_metadata
from pathlib import Path
from .inference_engine import InferenceEngine
from .processor import ClassificationProcessor

# Destructure constants
DEFAULT_BATCH_SIZE = CONSTANTS.CLASSIFICATION_BATCH_SIZE
DEFAULT_INPUT_SIZE = CONSTANTS.CLASSIFICATION_INPUT_SIZE
DEFAULT_INPUT_DEPTH = CONSTANTS.CLASSIFICATION_INPUT_DEPTH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Classify objects in vignette images using a trained CNN model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model --batch-size 64
  python -m modules.object_classification -i .../project/date/cycle/location/vignettes -m ./model -o ./output --input-size 64
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing vignette images to classify. If following the default output path, this should be <output_directory>/project/date/cycle/location/vignettes')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the classified object .csv and .txt files. The full path for the saved files will be <output_directory>/project/date/cycle/location/classified_objects.csv')
    parser.add_argument('-m', '--model', required=True,
                        help='Path to trained model directory containing the model.ckpt file')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for classification (default: 32)')
    parser.add_argument('--input-size', type=int, default=DEFAULT_INPUT_SIZE,
                        help='Input image size for the model (default: 50)')
    parser.add_argument('--input-depth', type=int, default=DEFAULT_INPUT_DEPTH,
                        help='Input image depth (channels) for the model (default: 1)')
    
    args = parser.parse_args()
    
    try:
        # Parse the vignette metadata from the input path and file name.
        metadata = parse_vignette_metadata(Path(args.input))

        # Pass the CLI arguments and metadata for validation, before running the classification.
        classification_data = validate_arguments(**vars(args), **metadata)

        # Initialize the inference engine and classification processor
        inference_engine = InferenceEngine(classification_data)
        processor = ClassificationProcessor()

        run_classification(classification_data, inference_engine, processor)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 