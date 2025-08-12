from typing import List, Tuple
from dataclasses import dataclass
from modules.common.constants import CONSTANTS

@dataclass
class ProcessedRegion:
    """Holds data for a single processed region."""
    region_id: int
    region_extents: Tuple[int, int, int, int]
    region_data: dict

# Destructured CONSTANTS for cleaner readability
MAX_ECCENTRICITY = CONSTANTS.MAX_ECCENTRICITY
MAX_MEAN_INTENSITY = CONSTANTS.MAX_MEAN_INTENSITY
MIN_MAJOR_AXIS_LENGTH = CONSTANTS.MIN_MAJOR_AXIS_LENGTH
MAX_MIN_INTENSITY = CONSTANTS.MAX_MIN_INTENSITY
SMALL_OBJECT_THRESHOLD = CONSTANTS.SMALL_OBJECT_THRESHOLD
MEDIUM_OBJECT_THRESHOLD = CONSTANTS.MEDIUM_OBJECT_THRESHOLD
LARGE_OBJECT_PADDING = CONSTANTS.LARGE_OBJECT_PADDING
SMALL_OBJECT_PADDING = CONSTANTS.SMALL_OBJECT_PADDING
MEDIUM_OBJECT_PADDING = CONSTANTS.MEDIUM_OBJECT_PADDING

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