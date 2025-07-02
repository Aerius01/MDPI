import numpy as np
from typing import List
from .config import DetectionConfig

class OverlapCorrector:
    """Handles depth overlap correction calculations."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def calculate_corrections(self, img_names: List[str]) -> np.ndarray:
        """Calculate pixel overlaps for depth correction."""
        depths = np.array([float(name.split('_')[0]) for name in img_names])
        image_bottom_depths = depths * self.config.depth_multiplier + self.config.image_height_cm
        image_top_depths = depths * self.config.depth_multiplier
        
        overlaps = np.zeros(len(depths))
        overlaps[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
        
        return np.round((overlaps / self.config.image_height_cm) * self.config.image_height_pixels).astype(int)
    
    def apply_corrections(self, img_bins: List[np.ndarray], img_names: List[str]) -> List[np.ndarray]:
        """Apply depth overlap correction to binary images."""
        pixel_overlaps = self.calculate_corrections(img_names)
        return [
            np.where(np.arange(img_bin.shape[0])[:, None] < pixel_overlap, 
                    self.config.threshold_max, img_bin)
            for img_bin, pixel_overlap in zip(img_bins, pixel_overlaps)
        ] 