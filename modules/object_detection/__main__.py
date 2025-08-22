#!/usr/bin/env python3
"""
Command line interface for object detection module.
Usage: python -m modules.object_detection [options]
"""

import argparse
import sys
import numpy as np
import cv2
from tqdm import tqdm

from modules.common.constants import CONSTANTS
from .detection_data import validate_arguments, DetectionData
from .detector import Detector, process_vignette
from .output_handler import OutputHandler

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE = CONSTANTS.BATCH_SIZE
CSV_EXTENSION = CONSTANTS.CSV_EXTENSION
CSV_SEPARATOR = CONSTANTS.CSV_SEPARATOR

# Filtering constants
THRESHOLD_VALUE = CONSTANTS.THRESHOLD_VALUE
THRESHOLD_MAX = CONSTANTS.THRESHOLD_MAX
MIN_OBJECT_SIZE = CONSTANTS.MIN_OBJECT_SIZE
MAX_OBJECT_SIZE = CONSTANTS.MAX_OBJECT_SIZE

# Region-oriented constants
MAX_ECCENTRICITY = CONSTANTS.MAX_ECCENTRICITY
MAX_MEAN_INTENSITY = CONSTANTS.MAX_MEAN_INTENSITY
MIN_MAJOR_AXIS_LENGTH = CONSTANTS.MIN_MAJOR_AXIS_LENGTH
MAX_MIN_INTENSITY = CONSTANTS.MAX_MIN_INTENSITY
SMALL_OBJECT_THRESHOLD = CONSTANTS.SMALL_OBJECT_THRESHOLD
MEDIUM_OBJECT_THRESHOLD = CONSTANTS.MEDIUM_OBJECT_THRESHOLD
LARGE_OBJECT_PADDING = CONSTANTS.LARGE_OBJECT_PADDING
SMALL_OBJECT_PADDING = CONSTANTS.SMALL_OBJECT_PADDING
MEDIUM_OBJECT_PADDING = CONSTANTS.MEDIUM_OBJECT_PADDING

def run_detection(
    data: DetectionData,
    detector: Detector,
    output_handler: OutputHandler
):
    """
    Main function to run the object detection process.
    """
    image_paths = data.flatfield_img_paths
    print(f"[DETECTION]: Found {len(image_paths)} flatfielded images")

    num_batches = int(np.ceil(len(image_paths) / detector.batch_size))
    print(f"[DETECTION]: Performing object detection in {num_batches} batches...")
    
    all_region_data = []
    output_count = 0
    
    for i in tqdm(range(0, len(image_paths), detector.batch_size), desc='[DETECTION]'):
        batch_end = i + detector.batch_size
        batch_image_paths = image_paths[i:batch_end]

        batch_images, batch_binary_images = detector.load_and_threshold_images(batch_image_paths)
        mapped_regions_batch = detector.detect_objects(batch_images, batch_binary_images, batch_image_paths)

        for mapped_region in mapped_regions_batch:
            process_vignette_generator = process_vignette(mapped_region, data.output_path)
            for region_data, vignette_img, vignette_path in process_vignette_generator:
                all_region_data.append(region_data)
                cv2.imwrite(vignette_path, vignette_img)
                output_count += 1
           
    combined_df = output_handler.create_dataframe(all_region_data, data)
    output_handler.save_dataframe(combined_df, data.output_path)

    print(f"[DETECTION]: Processing completed successfully!")
    print(f"[DETECTION]: {output_count} vignettes saved to {data.output_path}")

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
    parser.add_argument('-d', '--depth-profiles', required=True,
                        help='The path to the depth profiles .csv file.')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the detected object vignettes (individual images of detected objects). The full path for the saved images will be <output_directory>/project/date/cycle/location/vignettes')

    args = parser.parse_args()
    
    try:
        print(f"[DETECTION]: Loading images from {args.input}")
        detection_data = validate_arguments(args)

        detector_config = {
            'threshold_value': THRESHOLD_VALUE,
            'threshold_max': THRESHOLD_MAX,
            'min_object_size': MIN_OBJECT_SIZE,
            'max_object_size': MAX_OBJECT_SIZE,
            'max_eccentricity': MAX_ECCENTRICITY,
            'max_mean_intensity': MAX_MEAN_INTENSITY,
            'min_major_axis_length': MIN_MAJOR_AXIS_LENGTH,
            'max_min_intensity': MAX_MIN_INTENSITY,
            'small_object_threshold': SMALL_OBJECT_THRESHOLD,
            'medium_object_threshold': MEDIUM_OBJECT_THRESHOLD,
            'large_object_padding': LARGE_OBJECT_PADDING,
            'small_object_padding': SMALL_OBJECT_PADDING,
            'medium_object_padding': MEDIUM_OBJECT_PADDING,
            'batch_size': BATCH_SIZE
        }
        detector = Detector(**detector_config)

        output_handler_config = {
            'csv_extension': CSV_EXTENSION,
            'csv_separator': CSV_SEPARATOR
        }
        output_handler = OutputHandler(**output_handler_config)

        run_detection(detection_data, detector, output_handler)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1) 