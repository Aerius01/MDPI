from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass(frozen=True)
class ProcessingConstants:
    """Centralized processing constants grouped by module for readability.

    The attributes are organized in the same order as the pipeline:
    depth profiling → flatfielding → object detection → object classification.
    Shared items come first. Names are preserved to avoid breaking imports.
    """

    # ── Shared/common (used across modules) ────────────────────────────────────
    BATCH_SIZE: int = 10  # How many images to process at a given time?

    # File formats
    JPEG_EXTENSION: str = '.jpeg'
    TIFF_EXTENSION: str = '.tiff'
    CSV_EXTENSION: str = '.csv'

    # Shared image geometry used for depth calculations
    # The MDPI is placed horizontally as it moves through the water column,
    # meaning that images are vertical. 4.3 cm is the height of the field of view.
    IMAGE_HEIGHT_CM: float = 4.3
    IMAGE_HEIGHT_PIXELS: int = 2048

    # ── Duplicate detection utilities ─────────────────────────────────────────
    DUPLICATE_DETECTION_DISPLAY_SIZE: Tuple[int, int] = (500, 500)
    DUPLICATE_DETECTION_REMOVE: bool = False
    DUPLICATE_DETECTION_SHOW_MONTAGES: bool = True

    # ── Depth profiling module ─────────────────────────────────────────────────
    # Both cameras use semicolon as the separator
    CSV_SEPARATOR: str = ';'  # How are columns delineated?


    # Old camera format defaults (used as fallback when auto-detection fails)
    CSV_HEADER_ROW: int = 5   # Row index of the header (0-based)
    CSV_SKIPFOOTER: int = 1   # How many rows to skip after the footer?
    OLD_FORMAT_TIME_COLUMN_SEARCH: str = "Time SN:"  # Old format time column search term
    OLD_FORMAT_DEPTH_COLUMN_SEARCH: str = "bar"  # Old format depth column search term (pressure in bar)
    PRESSURE_SENSOR_DEPTH_MULTIPLIER: float = 10.0  # Convert pressure sensor values to depth (meters)

    # New camera format constants
    NEW_FORMAT_HEADER_ROW: int = 0  # New format has single header line
    NEW_FORMAT_SKIPFOOTER: int = 0  # New format has no footer to skip
    NEW_FORMAT_TIME_COLUMN_NAME: str = "Date-Time"  # New format time column name
    NEW_FORMAT_DEPTH_COLUMN_NAME: str = "Depth(m)"  # New format depth column name
    NEW_FORMAT_PRESSURE_MULTIPLIER: float = 1.0  # New format already in meters

    # Format-independent constants
    OVERLAP_CORRECTION_DEPTH_MULTIPLIER: int = 100  # Depth overlap correction

    # ── Flatfielding module ────────────────────────────────────────────────────
    # Empirical flatfielding factor → set by Tim W. & Jens N.
    NORMALIZATION_FACTOR: int = 235

    # ── Object detection module ────────────────────────────────────────────────
    # Thresholding
    THRESHOLD_VALUE: int = 190
    THRESHOLD_MAX: int = 255
    
    # Object size filtering --> the pixel areal limit required for us to 
    # identify the object, will need to change with MDPI
    MIN_OBJECT_SIZE: int = 75
    MAX_OBJECT_SIZE: int = 5000  # potentially obsolete

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

    # ── Object classification module ───────────────────────────────────────────
    CLASSIFICATION_BATCH_SIZE: int = 32
    CLASSIFICATION_INPUT_SIZE: int = 50
    CLASSIFICATION_INPUT_DEPTH: int = 1
    CLASSIFICATION_CATEGORIES: List[str] = field(
        default_factory=lambda: ['cladocera', 'copepod', 'junk', 'rotifer']
    )

# Global instance for easy access
CONSTANTS = ProcessingConstants()