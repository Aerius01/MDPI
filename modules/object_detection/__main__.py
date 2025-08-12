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
from .config import DetectionConfig
from .detector import ObjectDetector

def main():
    parser = argparse.ArgumentParser(
        description='Detect objects in images and extract vignettes.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.object_detection -i ./flatfielded
  python -m modules.object_detection -i ./flatfielded -o ./output --threshold-value 180
  python -m modules.object_detection -i ./flatfielded -o ./output --min-object-size 100 --max-object-size 3000
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images for object detection')
    parser.add_argument('-d', '--depth-mapping', required=True,
                        help='Path to the depth mapping CSV file')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the detected object vignettes. The full path will be <output_directory>/project/date/cycle/location/vignettes')
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
        # Get image paths from input directory
        print(f"[DETECTION]: Loading images from {args.input}")
        input_path = Path(args.input)
        metadata = parse_flatfield_metadata_from_directory(input_path)
        image_paths = metadata['raw_img_paths']
        print(f"[DETECTION]: Found {len(image_paths)} flatfielded images")

        # Load depth mapping
        print(f"[DETECTION]: Loading depth mapping from {args.depth_mapping}")
        depth_df = pd.read_csv(args.depth_mapping)
        depth_map = {}
        # Assuming the CSV has 'image_path' and 'depth' columns.
        # 'image_path' should contain the filename or full path of the RAW image.
        for _, row in depth_df.iterrows():
            # Extracts 'HHMMSSmmm_replicate' from raw image path to use as a key,
            # matching the stem of the flat-fielded image filenames.
            raw_path_stem = Path(row['image_path']).stem
            parts = raw_path_stem.split('_')
            if len(parts) >= 2:
                key = f"{parts[-2]}_{parts[-1]}"
                depth_map[key] = row['depth']
        print(f"[DETECTION]: Loaded {len(depth_map)} depth mappings.")

        # Validate and create output path
        output_dir = CommonCLI.validate_output_path(args.output)
        date_str = metadata["recording_start_date"].strftime("%Y%m%d")
        output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], "vignettes")
        os.makedirs(output_path, exist_ok=True)
        
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
        
        # Initialize the detector
        detector = ObjectDetector(config)

        # Step 1: Load and threshold all images
        print(f"[DETECTION]: Loading and thresholding...")
        images, binary_images = detector._load_and_threshold_images(image_paths)
        
        # Step 2: Process in batches
        num_batches = int(np.ceil(len(image_paths) / config.batch_size))
        print(f"[DETECTION]: Performing object detection in {num_batches} batches...")
        
        all_region_data = []
        all_output_files = []
        
        for i in tqdm(range(0, len(image_paths), config.batch_size), desc='[DETECTION]'):
            batch_end = i + config.batch_size
            batch_images = images[i:batch_end]
            batch_binary_images = binary_images[i:batch_end]
            batch_image_paths = image_paths[i:batch_end]

            # Process the batch to get regions mapped to each image
            mapped_regions_batch = detector.detect_objects(batch_images, batch_binary_images, batch_image_paths)

            # Unpack the results, save vignettes, and collect data for CSV
            for mapped_region in mapped_regions_batch:
                img_name = Path(mapped_region.source_image_path).stem
                
                # Get depth for this image
                depth = depth_map.get(img_name)
                if depth is None:
                    raise ValueError(f"No depth mapping found for image: {img_name}. Check your depth mapping file.")
                
                for region in mapped_region.processed_regions:
                    # Add image-specific info to the region data
                    region.data['FileName'] = img_name
                    region.data['depth'] = depth
                    all_region_data.append(region.data)
                    
                    # Crop and save vignette
                    minr, maxr, minc, maxc = region.extents
                    vignette = mapped_region.source_image[minr:maxr, minc:maxc]
                    
                    vignette_filename = f"{img_name}_vignette_{region.id}.png"
                    vignette_path = os.path.join(output_path, vignette_filename)
                    cv2.imwrite(vignette_path, vignette)
                    all_output_files.append(vignette_path)

        # Create a combined DataFrame from all processed regions
        if all_region_data:
            combined_df = pd.DataFrame(all_region_data)
            # Reorder columns to have FileName and depth first
            cols = ['FileName', 'depth'] + [col for col in combined_df.columns if col not in ['FileName', 'depth']]
            combined_df = combined_df[cols]
        else:
            combined_df = pd.DataFrame()

        # Get project, date, time, location from metadata for filename
        project = metadata["project"]
        date = metadata["recording_start_date"].strftime("%Y%m%d")
        time = metadata["recording_start_time"].strftime("%H%M%S")
        location = metadata["location"]

        # Save results if any objects were detected
        if not combined_df.empty:
            print(f"[DETECTION]: Detection completed successfully! Total objects detected: {len(combined_df)}")
            # Save CSV file
            csv_output_file = os.path.join(output_path, f'objectMeasurements_{project}_{date}_{time}_{location}{config.csv_extension}')
            combined_df.to_csv(csv_output_file, sep=config.csv_separator, index=False)
            
            # Save text file (space-separated)
            txt_output_file = os.path.join(output_path, f'objectMeasurements_{project}_{date}_{time}_{location}.txt')
            combined_df.to_csv(txt_output_file, sep=' ', index=False)
            
            print(f"[DETECTION]: Saved results to {csv_output_file} and {txt_output_file}")
        else:
            print(f"[DETECTION]: Detection completed. No objects detected.")

        print(f"[DETECTION]: Processing completed successfully!")
        if all_output_files:
            print(f"[DETECTION]: {len(all_output_files)} vignettes saved to {output_path}")
        else:
            print(f"[DETECTION]: No files were processed")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 