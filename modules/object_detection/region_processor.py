import cv2
import pandas as pd
from pathlib import Path
import numpy as np
from typing import List, Tuple
from .config import DetectionConfig
import os

# Column definitions for consistent DataFrame creation
MEASUREMENT_COLUMNS = [
    'Filename', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
    'Orientation', 'EquivDiameter', 'Solidity', 'Extent', 'MaxIntensity', 
    'MeanIntensity', 'MinIntensity', 'Perimeter'
]

class RegionProcessor:
    """Handles region filtering and processing."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def filter_regions(self, regions: List) -> List:
        """Apply all filtering criteria to regions and return valid ones."""
        if not regions:
            return []
        
        return [
            r for r in regions
            if (r.eccentricity < self.config.max_eccentricity and
                r.mean_intensity < self.config.max_mean_intensity and
                r.major_axis_length > self.config.min_major_axis_length and
                r.min_intensity < self.config.max_min_intensity)
        ]
    
    def calculate_crop_padding(self, major_axis_length: float) -> int:
        """Calculate appropriate padding based on object size."""
        if major_axis_length < self.config.small_object_threshold:
            return self.config.small_object_padding
        elif major_axis_length < self.config.medium_object_threshold:
            return self.config.medium_object_padding
        else:
            return self.config.large_object_padding
    
    def extract_and_save_region(self, region, image: np.ndarray, img_name: str, 
                               index: int, output_path: str) -> Tuple[List, str]:
        """Extract region data and save cropped image."""
        # Extract region measurements
        region_data = [
            f"{img_name}_{index}",
            region.area,
            region.major_axis_length,
            region.minor_axis_length,
            region.eccentricity,
            region.orientation,
            region.equivalent_diameter,
            region.solidity,
            region.extent,
            region.max_intensity,
            region.mean_intensity,
            region.min_intensity,
            region.perimeter
        ]
        
        # Calculate crop boundaries
        row, col = int(region.centroid[0]), int(region.centroid[1])
        padding = self.calculate_crop_padding(region.major_axis_length)
        
        minr = max(0, row - padding)
        minc = max(0, col - padding)
        maxr = min(image.shape[0], row + padding)
        maxc = min(image.shape[1], col + padding)
        
        # Save cropped region
        crop_img = image[minr:maxr, minc:maxc]
        output_file = os.path.join(output_path, f'{img_name}_{index}{self.config.jpeg_extension}')
        cv2.imwrite(output_file, crop_img)
        
        return region_data, output_file
    
    def process_regions(self, regions: List, image: np.ndarray, img_name: str, 
                       output_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Process regions: filter, extract data, and save crops."""
        valid_regions = self.filter_regions(regions)
        if not valid_regions:
            return pd.DataFrame(columns=MEASUREMENT_COLUMNS), []
        
        # Extract data and save crops for all valid regions
        data_list = []
        output_paths = []
        for i, region in enumerate(valid_regions):
            region_data, output_file = self.extract_and_save_region(region, image, img_name, i, output_path)
            data_list.append(region_data)
            output_paths.append(output_file)
        
        return pd.DataFrame(data_list, columns=MEASUREMENT_COLUMNS), output_paths 