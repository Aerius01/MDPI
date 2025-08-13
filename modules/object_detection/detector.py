import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from modules.common.constants import CONSTANTS

# Destructured CONSTANTS for cleaner readability
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

def load_and_threshold_images(image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load images and apply thresholding."""
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    
    # Apply thresholding
    img_bins = [cv2.threshold(img, THRESHOLD_VALUE, THRESHOLD_MAX, cv2.THRESH_BINARY_INV)[1] 
                for img in images]
    
    return images, img_bins

def _apply_size_filtering(label_imgs: List[np.ndarray]) -> List[np.ndarray]:
    """Apply size-based filtering to labeled images."""
    return [
        remove_small_objects(label_img > 0, min_size=MIN_OBJECT_SIZE) & 
        ~remove_small_objects(label_img > 0, min_size=MAX_OBJECT_SIZE)
        for label_img in label_imgs
    ]

def detect_objects(images: List[np.ndarray], binary_images: List[np.ndarray], image_paths: List[str]) -> List[MappedImageRegions]:
    """Process a batch of images to detect and analyze objects."""
    # Perform object detection and size filtering
    labeled_images = [label(img) for img in binary_images]
    filtered_images = _apply_size_filtering(labeled_images)
    
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

def filter_regions(regions: List) -> List:
    """Apply all filtering criteria to regions and return valid ones."""
    if not regions:
        return []
    
    return [
        r for r in regions
        if (r.eccentricity < MAX_ECCENTRICITY and
            r.mean_intensity < MAX_MEAN_INTENSITY and
            r.major_axis_length > MIN_MAJOR_AXIS_LENGTH and
            r.min_intensity < MAX_MIN_INTENSITY)
    ]

def calculate_crop_padding(major_axis_length: float) -> int:
    """Calculate appropriate padding based on object size."""
    if major_axis_length < SMALL_OBJECT_THRESHOLD:
        return SMALL_OBJECT_PADDING
    elif major_axis_length < MEDIUM_OBJECT_THRESHOLD:
        return MEDIUM_OBJECT_PADDING
    else:
        return LARGE_OBJECT_PADDING

def process_regions(regions: List, image_shape: Tuple[int, int]) -> List[ProcessedRegion]:
    """Process regions: filter, extract data, and save crops."""
    valid_regions = filter_regions(regions)
    if not valid_regions:
        return []
    
    processed_regions: List[ProcessedRegion] = []
    for i, region in enumerate(valid_regions):
        
        # Calculate crop boundaries
        row, col = int(region.centroid[0]), int(region.centroid[1])
        padding = calculate_crop_padding(region.major_axis_length)
        
        minr = max(0, row - padding)
        minc = max(0, col - padding)
        maxr = min(image_shape[0], row + padding)
        maxc = min(image_shape[1], col + padding)

        # Extract region measurements
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