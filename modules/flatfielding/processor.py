import cv2
import numpy as np
from typing import List, Tuple

class FlatfieldProcessor:
    """Handles flatfielding operations on image groups."""
    
    def __init__(self):
        pass
    
    def calculate_average_image(self, image_group: List[str]) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
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
    
    def flatfield_image(self, base_image: np.ndarray, average_image: np.ndarray, normalization_factor: int) -> np.ndarray:
        """
        Performs flatfielding on a single image or a batch of images.
        Accepts a base image and an average image, and then returns the flatfielded image.
        """
        flatfielded_image = np.divide(base_image, average_image) * normalization_factor
        flatfielded_image = np.clip(flatfielded_image, 0, 255).astype('uint8')
        
        return flatfielded_image