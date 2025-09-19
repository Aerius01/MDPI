import numpy as np
import cv2
from tqdm import tqdm

from .detection_data import DetectionData
from .detector import Detector, process_vignette
from .output_handler import OutputHandler


def run_detection(
    data: DetectionData,
    detector: Detector,
    output_handler: OutputHandler
):
    """
    Run the object detection process over flatfielded images and save outputs.
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


