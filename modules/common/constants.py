import re
from dateutil import relativedelta
from dataclasses import dataclass

def get_timestep_from_rate(capture_rate: float) -> relativedelta.relativedelta:
    """Calculate timestep from capture rate in Hz."""
    if capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")
    return relativedelta.relativedelta(microseconds=1/capture_rate*1000000)

# Pre-compile regex for efficiency - matches datetime pattern in image filenames
BASE_FILENAME_PATTERN = re.compile(r'(\d{8}_\d{6}\d{3})_(\d+)\.') 

def get_image_sort_key(path: str) -> int:
    """Extract sort key from image path based on sequence number in filename.
    
    Args:
        path: Path to image file
        
    Returns:
        Sequence number from filename, or 0 if no match found
    """
    import os
    filename = os.path.basename(path)
    match = BASE_FILENAME_PATTERN.search(filename)
    return int(match.group(2)) if match else 0

@dataclass(frozen=True)
class ProcessingConstants:
    """Centralized processing constants."""
    # Batch processing
    BATCH_SIZE: int = 10
    
    # Empirical flatfielding factor --> set by Tim W. & Jens N.
    NORMALIZATION_FACTOR: int = 235
    
    # File formats
    JPEG_EXTENSION: str = '.jpeg'
    TIFF_EXTENSION: str = '.tiff'
    CSV_EXTENSION: str = '.csv'
    
    # CSV processing
    CSV_SEPARATOR: str = ';'
    CSV_HEADER_ROW: int = 6
    CSV_SKIPFOOTER: int = 1
    
    # Depth profiling: convert depth values to meters
    DEPTH_MULTIPLIER: float = 10.0
    
    # Object detection thresholding
    THRESHOLD_VALUE: int = 190
    THRESHOLD_MAX: int = 255
    
    # Object size filtering --> the pixel areal limit required for us to identify the object, will need to change with MDPI
    MIN_OBJECT_SIZE: int = 75
    MAX_OBJECT_SIZE: int = 5000 # potentially obsolete
    
    # Region filtering criteria
    MAX_ECCENTRICITY: float = 0.97
    MAX_MEAN_INTENSITY: int = 130
    MIN_MAJOR_AXIS_LENGTH: int = 25
    MAX_MIN_INTENSITY: int = 65
    
    # Object cropping parameters
    SMALL_OBJECT_PADDING: int = 25
    MEDIUM_OBJECT_PADDING: int = 30
    LARGE_OBJECT_PADDING: int = 40
    SMALL_OBJECT_THRESHOLD: int = 40
    MEDIUM_OBJECT_THRESHOLD: int = 50
    
    # Depth overlap correction
    DEPTH_MULTIPLIER_CM: int = 100
    # The MDPI is placed horizontally as it moves through the water column, meaning that images are vertical. 4.3 cm is the height of the field of view.
    IMAGE_HEIGHT_CM: float = 4.3 
    IMAGE_HEIGHT_PIXELS: int = 2048

# Global instance for easy access
CONSTANTS = ProcessingConstants() 