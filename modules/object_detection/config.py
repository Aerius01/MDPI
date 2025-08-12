from dataclasses import dataclass
from modules.common.constants import CONSTANTS

@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    # Depth overlap correction
    depth_multiplier: int = CONSTANTS.OVERLAP_CORRECTION_DEPTH_MULTIPLIER
    image_height_cm: float = CONSTANTS.IMAGE_HEIGHT_CM
    image_height_pixels: int = CONSTANTS.IMAGE_HEIGHT_PIXELS
    
    # Image thresholding
    threshold_value: int = CONSTANTS.THRESHOLD_VALUE
    threshold_max: int = CONSTANTS.THRESHOLD_MAX
    
    # Object filtering
    min_object_size: int = CONSTANTS.MIN_OBJECT_SIZE
    max_object_size: int = CONSTANTS.MAX_OBJECT_SIZE
    
    # Region filtering
    max_eccentricity: float = CONSTANTS.MAX_ECCENTRICITY
    max_mean_intensity: int = CONSTANTS.MAX_MEAN_INTENSITY
    min_major_axis_length: int = CONSTANTS.MIN_MAJOR_AXIS_LENGTH
    max_min_intensity: int = CONSTANTS.MAX_MIN_INTENSITY
    
    # Object cropping
    small_object_padding: int = CONSTANTS.SMALL_OBJECT_PADDING
    medium_object_padding: int = CONSTANTS.MEDIUM_OBJECT_PADDING
    large_object_padding: int = CONSTANTS.LARGE_OBJECT_PADDING
    small_object_threshold: int = CONSTANTS.SMALL_OBJECT_THRESHOLD
    medium_object_threshold: int = CONSTANTS.MEDIUM_OBJECT_THRESHOLD
    
    # File operations
    csv_separator: str = CONSTANTS.CSV_SEPARATOR
    jpeg_extension: str = CONSTANTS.JPEG_EXTENSION
    csv_extension: str = CONSTANTS.CSV_EXTENSION
    batch_size: int = CONSTANTS.BATCH_SIZE 