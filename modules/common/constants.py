import re
from dateutil import relativedelta
from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class ProcessingConstants:
    """Centralized processing constants."""
    # Batch processing
    BATCH_SIZE: int = 10 # How many images to process at a given time?
    
    # Empirical flatfielding factor --> set by Tim W. & Jens N.
    NORMALIZATION_FACTOR: int = 235
    
    # File formats
    JPEG_EXTENSION: str = '.jpeg'
    TIFF_EXTENSION: str = '.tiff'
    CSV_EXTENSION: str = '.csv'
    
    # CSV processing --> currently set for reading the pressure sensor data .csv file
    CSV_SEPARATOR: str = ';' # How are columns delineated?
    CSV_HEADER_ROW: int = 6 # How many rows to skip before the header?
    CSV_COLUMNS: Tuple[int, int] = (0, 1) # Which columns to read?
    CSV_SKIPFOOTER: int = 1 # How many rows to skip after the footer?
    
    # Depth profiling: convert depth values to meters
    PRESSURE_SENSOR_DEPTH_MULTIPLIER: float = 10.0
    
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
    OVERLAP_CORRECTION_DEPTH_MULTIPLIER: int = 100
    # The MDPI is placed horizontally as it moves through the water column, meaning that images are vertical. 4.3 cm is the height of the field of view.
    IMAGE_HEIGHT_CM: float = 4.3 
    IMAGE_HEIGHT_PIXELS: int = 2048

# Global instance for easy access
CONSTANTS = ProcessingConstants()