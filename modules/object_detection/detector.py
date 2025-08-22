import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class ProcessedRegion:
    """Holds data for a single processed region."""
    region_id: int
    region_extents: Tuple[int, int, int, int]
    region_data: dict

@dataclass
class MappedImageRegions:
    """Holds the source image and its detected regions."""
    source_image_path: str
    source_image: np.ndarray
    processed_regions: List[ProcessedRegion]

@dataclass
class Detector:
    threshold_value: int
    threshold_max: int
    min_object_size: int
    max_object_size: int
    max_eccentricity: float
    max_mean_intensity: float
    min_major_axis_length: float
    max_min_intensity: float
    small_object_threshold: int
    medium_object_threshold: int
    large_object_padding: int
    small_object_padding: int
    medium_object_padding: int
    batch_size: int

    def load_and_threshold_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images and apply thresholding."""
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        
        img_bins = [cv2.threshold(img, self.threshold_value, self.threshold_max, cv2.THRESH_BINARY_INV)[1] 
                    for img in images]
        
        return images, img_bins

    def _apply_size_filtering(self, label_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Apply size-based filtering to labeled images."""
        return [
            remove_small_objects(label_img > 0, min_size=self.min_object_size) & 
            ~remove_small_objects(label_img > 0, min_size=self.max_object_size)
            for label_img in label_imgs
        ]

    def detect_objects(self, images: List[np.ndarray], binary_images: List[np.ndarray], image_paths: List[str]) -> List[MappedImageRegions]:
        """Process a batch of images to detect and analyze objects."""
        labeled_images = [label(img) for img in binary_images]
        filtered_images = self._apply_size_filtering(labeled_images)
        
        regions = [regionprops(label(img), image) for img, image in zip(filtered_images, images)]

        processed_regions_by_image = [self.process_regions(regions[i], images[i].shape) for i in range(len(images))]
        
        mapped_image_regions = [
            MappedImageRegions(
                source_image_path=image_paths[i],
                source_image=images[i],
                processed_regions=processed_regions_by_image[i]
            ) 
            for i in range(len(image_paths))
        ]
        
        return mapped_image_regions

    def filter_regions(self, regions: List) -> List:
        """Apply all filtering criteria to regions and return valid ones."""
        if not regions:
            return []
        
        return [
            r for r in regions
            if (r.eccentricity < self.max_eccentricity and
                r.mean_intensity < self.max_mean_intensity and
                r.major_axis_length > self.min_major_axis_length and
                r.min_intensity < self.max_min_intensity)
        ]

    def calculate_crop_padding(self, major_axis_length: float) -> int:
        """Calculate appropriate padding based on object size."""
        if major_axis_length < self.small_object_threshold:
            return self.small_object_padding
        elif major_axis_length < self.medium_object_threshold:
            return self.medium_object_padding
        else:
            return self.large_object_padding

    def process_regions(self, regions: List, image_shape: Tuple[int, int]) -> List[ProcessedRegion]:
        """Process regions: filter, extract data, and save crops."""
        valid_regions = self.filter_regions(regions)
        if not valid_regions:
            return []
        
        processed_regions: List[ProcessedRegion] = []
        for i, region in enumerate(valid_regions):
            
            row, col = int(region.centroid[0]), int(region.centroid[1])
            padding = self.calculate_crop_padding(region.major_axis_length)
            
            minr = max(0, row - padding)
            minc = max(0, col - padding)
            maxr = min(image_shape[0], row + padding)
            maxc = min(image_shape[1], col + padding)

            region_data = {
                'Area': region.area,
                'MajorAxisLength': region.major_axis_length,
                'MinorAxisLength': region.minor_axis_length,
                'Eccentricity': region.eccentricity,
                'Orientation': region.orientation,
                'EquivDiameter': region.equivalent_diameter,
                'Solidity': region.solidity,
                'Extent': region.extent,
                'MaxIntensity': region.max_intensity,
                'MeanIntensity': region.mean_intensity,
                'MinIntensity': region.min_intensity,
                'Perimeter': region.perimeter
            }

            processed_regions.append(
                ProcessedRegion(
                    region_id=i,
                    region_extents=(minr, maxr, minc, maxc),
                    region_data=region_data
                )
            )
        
        return processed_regions

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
    image_id = int(img_name.split('_')[1])
    for region in mapped_region.processed_regions:
        
        # Crop vignette
        minr, maxr, minc, maxc = region.region_extents
        vignette_img = mapped_region.source_image[minr:maxr, minc:maxc]
        
        # Construct vignette path
        vignette_filename = f"{img_name}_vignette_{region.region_id}.png"
        vignette_path = os.path.join(output_path, vignette_filename)
    
        # Add image-specific info to the region data
        region.region_data['FileName'] = vignette_filename
        region.region_data['replicate'] = region.region_id
        region.region_data['image_id'] = image_id
        
        yield region.region_data, vignette_img, vignette_path