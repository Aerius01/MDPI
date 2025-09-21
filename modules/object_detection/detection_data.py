from pathlib import Path
import pandas as pd
from modules.common.parser import _list_image_filenames
import os
from dataclasses import dataclass, fields
from types import SimpleNamespace
from modules.common.constants import CONSTANTS

# ─── O B J E C T · D E T E C T I O N · P A R A M E T E R S ──────────────────────────
# Thresholding values for image binarization
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

# Object cropping parameters in pixels
SMALL_OBJECT_PADDING: int = 25
MEDIUM_OBJECT_PADDING: int = 30
LARGE_OBJECT_PADDING: int = 40
SMALL_OBJECT_THRESHOLD: int = 40
MEDIUM_OBJECT_THRESHOLD: int = 50

@dataclass
class DetectionData:
    """
    A dataclass to hold all the necessary data for the object detection process.
    """
    input_path: Path
    output_path: str
    depth_profiles_df: pd.DataFrame
    
    # Metadata attributes
    flatfield_img_paths: list[str]
    
    # Detector configuration
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

    def __init__(self, **kwargs):
        """
        Initializes the DetectionData object by setting attributes from keyword arguments.
        """
        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])

def _validate_arguments(run_config: SimpleNamespace, depth_profiles_df: pd.DataFrame) -> DetectionData:
    """
    Processes and validates the command-line arguments.
    """
    flatfield_dir = os.path.join(run_config.output_root, CONSTANTS.FLATFIELD_DIR_NAME)
    input_path = Path(flatfield_dir)
    
    # Get flatfield-specific file paths
    filenames = _list_image_filenames(input_path)
    flatfield_img_paths = [str(input_path / filename) for filename in filenames]
    run_config.metadata['flatfield_img_paths'] = flatfield_img_paths

    output_path = os.path.join(Path(run_config.output_root), CONSTANTS.VIGNETTES_DIR_NAME)
    os.makedirs(output_path, exist_ok=True)
    
    return DetectionData(
        **run_config.metadata,
        input_path=input_path,
        output_path=output_path,
        depth_profiles_df=depth_profiles_df[['image_id', 'depth']],
        threshold_value=THRESHOLD_VALUE,
        threshold_max=THRESHOLD_MAX,
        min_object_size=MIN_OBJECT_SIZE,
        max_object_size=MAX_OBJECT_SIZE,
        max_eccentricity=MAX_ECCENTRICITY,
        max_mean_intensity=MAX_MEAN_INTENSITY,
        min_major_axis_length=MIN_MAJOR_AXIS_LENGTH,
        max_min_intensity=MAX_MIN_INTENSITY,
        small_object_threshold=SMALL_OBJECT_THRESHOLD,
        medium_object_threshold=MEDIUM_OBJECT_THRESHOLD,
        large_object_padding=LARGE_OBJECT_PADDING,
        small_object_padding=SMALL_OBJECT_PADDING,
        medium_object_padding=MEDIUM_OBJECT_PADDING,
        batch_size=CONSTANTS.BATCH_SIZE
    ) 