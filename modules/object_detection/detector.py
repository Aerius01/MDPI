import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Tuple
from .config import DetectionConfig
from .region_processor import process_regions, ProcessedRegion
from dataclasses import dataclass

@dataclass
class MappedImageRegions:
    """Holds the source image and its detected regions."""
    source_image_path: str
    source_image: np.ndarray
    processed_regions: List[ProcessedRegion]

class ObjectDetector:
    """Main class for object detection operations."""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
    
    def _parse_image_metadata(self, image_path: str) -> Tuple[str, str, str, str]:
        """Extract metadata from image filename."""
        filename = Path(image_path).stem
        _, project, date, time, location = filename.split('_')
        return project, date, time, location
    
    def _load_and_threshold_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images and apply thresholding."""
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        
        # Apply thresholding
        img_bins = [cv2.threshold(img, self.config.threshold_value, 
                                 self.config.threshold_max, cv2.THRESH_BINARY_INV)[1] 
                   for img in images]
        
        return images, img_bins
    
    def _apply_size_filtering(self, label_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Apply size-based filtering to labeled images."""
        return [
            remove_small_objects(label_img > 0, min_size=self.config.min_object_size) & 
            ~remove_small_objects(label_img > 0, min_size=self.config.max_object_size)
            for label_img in label_imgs
        ]
    
    def detect_objects(self, images: List[np.ndarray], binary_images: List[np.ndarray], image_paths: List[str]) -> List[MappedImageRegions]:
        """Process a batch of images to detect and analyze objects."""
        # Perform object detection and size filtering
        labeled_images = [label(img) for img in binary_images]
        filtered_images = self._apply_size_filtering(labeled_images)
        
        # A list of lists of regions, where each index corresponds to the regions for the image with the same index
        regions = [regionprops(label(img), image) for img, image in zip(filtered_images, images)]

        # A list of lists of ProcessedRegion objects, mapped to the corresponding image
        processed_regions_by_image = [process_regions(regions[i], images[i].shape) for i in range(len(images))]
        
        # Create a list of MappedImageRegions to explicitly link images with their regions
        mapped_image_regions = [
            MappedImageRegions(
                source_image_path=image_paths[i],
                source_image=images[i],
                processed_regions=processed_regions_by_image[i]
            ) 
            for i in range(len(image_paths))
        ]
        
        return mapped_image_regions