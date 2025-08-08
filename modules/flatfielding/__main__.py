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
from modules.common.cli_utils import CommonCLI
from .processor import FlatfieldProcessor
from pathlib import Path
from modules.common.parser import parse_metadata
from modules.common.constants import CONSTANTS

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE = CONSTANTS.BATCH_SIZE
NORMALIZATION_FACTOR = CONSTANTS.NORMALIZATION_FACTOR
IMAGE_EXTENSION = CONSTANTS.JPEG_EXTENSION

def main():
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
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the flatfielded images. The full path for the images will be <output_directory>/project/date/cycle/location/flatfielded_images')
    
    args = parser.parse_args()

    # Get image paths from input directory
    print(f"[FLATFIELDING]: Loading images from {args.input}")
    input_path = Path(args.input)
    metadata = parse_metadata(input_path)

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], 
                              "flatfielded_images")
    os.makedirs(output_path, exist_ok=True)

    print(f"[FLATFIELDING]: Found {len(metadata['raw_img_paths'])} images")
    
    try:
        # Calculate the average image
        print(f"[FLATFIELDING]: Calculating average image...")
        processor = FlatfieldProcessor()
        average_image, image_data = processor.calculate_average_image(metadata["raw_img_paths"])

        # Process the images in batches
        print(f"[FLATFIELDING]: Flatfielding {len(image_data)} images in batches of {BATCH_SIZE}...")

        success_count = 0
        recording_start_time_str = metadata['recording_start_time'].strftime("%H%M%S%f")[:-3]
        for i in tqdm(range(0, len(image_data), BATCH_SIZE), desc="Flatfielding batches"):
            batch_data = image_data[i:i + BATCH_SIZE]

            for image_path, image_array in batch_data:
                # Flatfield the image
                flatfielded_image = processor.flatfield_image(image_array, average_image, NORMALIZATION_FACTOR)

                # Save the flatfielded image
                base_filename = Path(image_path).stem
                replicate = base_filename.split('_')[-1]
                output_filename = f"{recording_start_time_str}_{replicate}{IMAGE_EXTENSION}"
                output_image_path = os.path.join(output_path, output_filename)

                image = Image.fromarray(flatfielded_image)
                image.save(output_image_path)
                
                success_count += 1
        
        print(f"[FLATFIELDING]: {success_count} files saved to {output_path}")
        print(f"[FLATFIELDING]: Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 