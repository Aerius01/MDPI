#!/usr/bin/env python3
"""
Command line interface for flatfielding module.
Usage: python3 -m modules.flatfielding [options]
"""

import argparse
import sys
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from dataclasses import dataclass
from modules.common.cli_utils import CommonCLI
from .flatfielding import calculate_average_image, flatfield_image
from pathlib import Path
from modules.common.parser import parse_metadata
from modules.common.constants import CONSTANTS

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE = CONSTANTS.BATCH_SIZE
NORMALIZATION_FACTOR = CONSTANTS.NORMALIZATION_FACTOR
IMAGE_EXTENSION = CONSTANTS.JPEG_EXTENSION

@dataclass
class ValidatedArguments:
    output_path: str
    metadata: dict
    overlap_map: dict

def process_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Get image paths from input directory
    print(f"[FLATFIELDING]: Loading images from {args.input}")
    input_path = Path(args.input)
    metadata = parse_metadata(input_path)

    # Load depth profiles to get pixel overlaps
    print(f"[FLATFIELDING]: Loading depth profiles from {args.depth_profiles}")
    depth_df = pd.read_csv(args.depth_profiles)
    # Create a mapping from absolute image path to pixel_overlap value
    overlap_map = pd.Series(depth_df.pixel_overlap.values, index=depth_df.image_path).to_dict()

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], 
                              "flatfielded_images")
    os.makedirs(output_path, exist_ok=True)

    return ValidatedArguments(output_path, metadata, overlap_map)

def save_flatfielded_image(
    flatfielded_image_array: np.ndarray,
    original_image_path: str,
    output_dir: str,
    recording_start_time_str: str,
    image_extension: str
):
    """Saves a single flatfielded image with a new filename."""
    base_filename = Path(original_image_path).stem
    replicate = base_filename.split('_')[-1]
    output_filename = f"{recording_start_time_str}_{replicate}{image_extension}"
    output_image_path = os.path.join(output_dir, output_filename)

    image = Image.fromarray(flatfielded_image_array)
    image.save(output_image_path)

def main(validated_arguments: ValidatedArguments):
    metadata = validated_arguments.metadata
    output_path = validated_arguments.output_path
    overlap_map = validated_arguments.overlap_map

    print(f"[FLATFIELDING]: Found {len(metadata['raw_img_paths'])} images")
    
    # Calculate the average image
    print(f"[FLATFIELDING]: Calculating average image...")
    average_image, image_data = calculate_average_image(metadata["raw_img_paths"])

    # Process the images in batches
    print(f"[FLATFIELDING]: Flatfielding {len(image_data)} images in batches of {BATCH_SIZE}...")

    success_count = 0
    recording_start_time_str = metadata['recording_start_time'].strftime("%H%M%S%f")[:-3]
    for i in tqdm(range(0, len(image_data), BATCH_SIZE), desc="Flatfielding batches"):
        batch_data = image_data[i:i + BATCH_SIZE]

        for image_path, image_array in batch_data:
            # Flatfield the image
            flatfielded_image = flatfield_image(image_array, average_image, NORMALIZATION_FACTOR)

            # Apply overlap correction using the absolute image path as the key
            abs_image_path = os.path.abspath(image_path)
            pixel_overlap = overlap_map.get(abs_image_path, 0)
            
            if pixel_overlap > 0:
                flatfielded_image[:pixel_overlap, :] = 0 # Black out the top rows

            # Save the flatfielded image
            save_flatfielded_image(
                flatfielded_image,
                image_path,
                output_path,
                recording_start_time_str,
                IMAGE_EXTENSION
            )
            
            success_count += 1
    
    print(f"[FLATFIELDING]: {success_count} files saved to {output_path}")
    print(f"[FLATFIELDING]: Processing completed successfully!")

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
                        help='Path to the depth profiles CSV file which includes pixel overlap data.')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the flatfielded images. The full path for the images will be <output_directory>/project/date/cycle/location/flatfielded_images')
    
    args = parser.parse_args()
    
    try:
        validated_arguments = process_arguments(args)
        main(validated_arguments)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 
