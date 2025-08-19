#!/usr/bin/env python3
"""
Command line interface for object detection module.
Usage: python -m modules.object_detection [options]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from modules.common.cli_utils import CommonCLI
from modules.common.parser import parse_flatfield_metadata_from_directory
from .detector import load_and_threshold_images, detect_objects
from modules.common.constants import CONSTANTS
from .detector import MappedImageRegions
from dataclasses import dataclass

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE= CONSTANTS.BATCH_SIZE
CSV_EXTENSION = CONSTANTS.CSV_EXTENSION
CSV_SEPARATOR = CONSTANTS.CSV_SEPARATOR

@dataclass
class ValidatedArguments:
    input_path: str
    output_path: str
    metadata: dict

def process_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Get image paths from input directory
    input_path = Path(args.input)
    metadata = parse_flatfield_metadata_from_directory(input_path)

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], "vignettes")
    os.makedirs(output_path, exist_ok=True)

    return ValidatedArguments(input_path, output_path, metadata)

def process_vignette(mapped_region: MappedImageRegions, output_path: str):
    """
    Prepare vignette data and yield it for each processed region.
    
    This function acts as a generator, yielding the region data, the vignette image,
    and the path to save the vignette for each detected object in a mapped region *AS* 
    the calling method also loops over the regions. In this way, the vignette data does 
    not need to be entirely processed and then stored in memory, but can instead be processed 
    and then saved as it is generated, immediately being released from memory. It's as though 
    the outer loop and this inner loop are connected and running synchronously.
    """
    img_name = Path(mapped_region.source_image_path).stem
    for region in mapped_region.processed_regions:
        # Add image-specific info to the region data
        region.region_data['FileName'] = img_name
        
        # Crop vignette
        minr, maxr, minc, maxc = region.region_extents
        vignette_img = mapped_region.source_image[minr:maxr, minc:maxc]
        
        # Construct vignette path
        vignette_filename = f"{img_name}_vignette_{region.region_id}.png"
        vignette_path = os.path.join(output_path, vignette_filename)
        
        yield region.region_data, vignette_img, vignette_path

def create_dataframe(data_list: list) -> pd.DataFrame:
    # Create a combined DataFrame from all processed regions
    if not data_list:
        return pd.DataFrame()
    
    # Create dataframe and reorder columns to have FileName first
    combined_df = pd.DataFrame(data_list)
    cols = ['FileName'] + [col for col in combined_df.columns if col not in ['FileName']]
    combined_df = combined_df[cols]

    return combined_df

def save_dataframe(metadata: dict, combined_df: pd.DataFrame, output_path: str) -> None:
    """Save detection results to CSV and text files."""

    if combined_df.empty:
        print(f"[DETECTION]: Detection completed. No objects detected.")
        return

    # Get project, date, time, location from metadata for filename
    project = metadata["project"]
    date = metadata["recording_start_date"].strftime("%Y%m%d")
    time = metadata["recording_start_time"].strftime("%H%M%S")
    location = metadata["location"]

    # Save results
    print(f"[DETECTION]: Detection completed successfully! Total objects detected: {len(combined_df)}")
    # Save CSV file
    csv_output_file = os.path.join(output_path, f'objectMeasurements_{project}_{date}_{time}_{location}{CSV_EXTENSION}')
    combined_df.to_csv(csv_output_file, sep=CSV_SEPARATOR, index=False)
    print(f"[DETECTION]: Saved results to {csv_output_file}")
    
    # Save text file (space-separated)
    txt_output_file = os.path.join(output_path, f'objectMeasurements_{project}_{date}_{time}_{location}.txt')
    combined_df.to_csv(txt_output_file, sep=' ', index=False)
    print(f"[DETECTION]: Copied results to {txt_output_file}")

def main(validated_arguments: ValidatedArguments):
    image_paths = validated_arguments.metadata['raw_img_paths']
    print(f"[DETECTION]: Found {len(image_paths)} flatfielded images")

    # Step 2: Process images in batches to reduce peak memory usage
    num_batches = int(np.ceil(len(image_paths) / BATCH_SIZE))
    print(f"[DETECTION]: Performing object detection in {num_batches} batches...")
    
    all_region_data = []
    output_count = 0
    
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc='[DETECTION]'):
        batch_end = i + BATCH_SIZE
        batch_image_paths = image_paths[i:batch_end]

        # Load and threshold only the current batch
        batch_images, batch_binary_images = load_and_threshold_images(batch_image_paths)

        # Process the batch to get regions mapped to each image
        mapped_regions_batch = detect_objects(batch_images, batch_binary_images, batch_image_paths)

        # Unpack the results, save vignettes, and collect data for CSV
        for mapped_region in mapped_regions_batch:
            process_vignette_generator = process_vignette(mapped_region, validated_arguments.output_path)
            for region_data, vignette_img, vignette_path in process_vignette_generator:
                all_region_data.append(region_data)
                cv2.imwrite(vignette_path, vignette_img)
                output_count += 1
           
    # Step 3: Create a combined DataFrame from all processed regions and then save it
    combined_df = create_dataframe(all_region_data)
    save_dataframe(validated_arguments.metadata, combined_df, validated_arguments.output_path)

    print(f"[DETECTION]: Processing completed successfully!")
    print(f"[DETECTION]: {output_count} vignettes saved to {validated_arguments.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect objects in images and extract vignettes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_detection -i ./project/date/cycle/location/flatfielded
  python -m modules.object_detection -i ./project/date/cycle/location/flatfielded -o ./output
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='The path to the directory containing the flatfielded images for object detection. If following the default output path, this should be <output_directory>/project/date/cycle/location/flatfielded')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the detected object vignettes (individual images of detected objects). The full path for the saved images will be <output_directory>/project/date/cycle/location/vignettes')
    
    args = parser.parse_args()
    
    try:
        # Step 1: Process arguments
        print(f"[DETECTION]: Loading images from {args.input}")
        validated_arguments = process_arguments(args)
        main(validated_arguments)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 