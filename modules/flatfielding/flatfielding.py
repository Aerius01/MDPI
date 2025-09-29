import cv2
import numpy as np
from typing import List, Tuple
from PIL import Image
import os
from pathlib import Path
from .flatfielding_data import FlatfieldingData

def calculate_average_image(image_group: List[str]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    Calculates the average image from a group of images.
    Also returns the original images coupled with their paths.
    """
    image_data = [(path, cv2.imread(path, cv2.IMREAD_GRAYSCALE)) for path in image_group]
    image_arrays = np.array([img[1] for img in image_data])

    average_image = np.average(image_arrays, axis=0).astype('uint8')

    # Prevent division by zero errors by replacing 0s with 1s in the average image
    average_image[average_image == 0] = 1

    return average_image, image_data

def flatfield_image(base_image: np.ndarray, average_image: np.ndarray, normalization_factor: int) -> np.ndarray:
    """
    Performs flatfielding on a single image or a batch of images.
    Accepts a base image and an average image, and then returns the flatfielded image.
    """
    flatfielded_image = np.divide(base_image, average_image) * normalization_factor
    flatfielded_image = np.clip(flatfielded_image, 0, 255).astype('uint8')
    
    return flatfielded_image
    
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

def flatfield_images(data: FlatfieldingData):
    """
    Orchestrates the flatfielding process.
    """
    print(f"[FLATFIELDING]: Found {len(data.metadata['raw_img_paths'])} images")
    
    # Calculate the average image
    print(f"[FLATFIELDING]: Calculating average image...")
    average_image, image_data = calculate_average_image(data.metadata["raw_img_paths"])

    # Process the images in batches
    print(f"[FLATFIELDING]: Flatfielding {len(image_data)} images in batches of {data.batch_size}...")

    success_count = 0
    total_images = len(image_data)
    recording_start_time_str = data.metadata['recording_start_time'].strftime("%H%M%S%f")[:-3]
    
    for i in range(0, total_images, data.batch_size):
        batch_data = image_data[i:i + data.batch_size]

        for image_path, image_array in batch_data:
            # Flatfield the image
            flatfielded_image = flatfield_image(image_array, average_image, data.normalization_factor)

            # Apply overlap correction using the absolute image path as the key
            abs_image_path = os.path.abspath(image_path)
            pixel_overlap = data.overlap_map.get(abs_image_path, 0)
            
            if pixel_overlap > 0:
                flatfielded_image[:pixel_overlap, :] = 255 # White out the top rows

            # Save the flatfielded image
            save_flatfielded_image(
                flatfielded_image,
                image_path,
                data.output_path,
                recording_start_time_str,
                data.image_extension
            )
            
            success_count += 1
        
        # Print progress after each batch
        progress = min(i + data.batch_size, total_images)
        print(f"[PROGRESS] {progress}/{total_images}")

    # Hardcode a final, completed progress bar to bypass async issues
    bar = f"[{'#' * 40}]"
    total_str = str(total_images)
    progress_bar = f"[FLATFIELDING]: {bar} 100% | {total_str}/{total_str}"
    print(progress_bar, flush=True)

    print(f"[FLATFIELDING]: {success_count} files saved to {data.output_path}")