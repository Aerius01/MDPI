import argparse
from pathlib import Path
import pandas as pd
from modules.common.fs_utils import ensure_dir
from .utils import parse_flatfield_metadata_from_directory
import os
from dataclasses import dataclass, fields

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

OUTPUT_CSV_SEPARATOR = ';'

@dataclass
class DetectionData:
    """
    A dataclass to hold all the necessary data for the object detection process.
    """
    input_path: Path
    output_path: str
    depth_profiles_df: pd.DataFrame
    
    # Metadata attributes
    project: str
    recording_start_date: str
    cycle: str
    location: str
    flatfield_img_paths: list[str]

    def __init__(self, **kwargs):
        """
        Initializes the DetectionData object by setting attributes from keyword arguments.
        """
        for field in fields(self):
            if field.name in kwargs:
                setattr(self, field.name, kwargs[field.name])

def validate_arguments(args: argparse.Namespace) -> DetectionData:
    """
    Processes and validates the command-line arguments.
    """
    input_path = Path(args.input)
    metadata = parse_flatfield_metadata_from_directory(input_path)

    output_dir = ensure_dir(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], "vignettes")
    os.makedirs(output_path, exist_ok=True)

    depth_profiles_path = Path(args.depth_profiles)
    if not depth_profiles_path.is_file():
        raise FileNotFoundError(f"Depth profiles CSV file not found at: {depth_profiles_path}")
    
    depth_profiles_df = pd.read_csv(depth_profiles_path, sep=None, engine='python')
    required_columns = ['image_id', 'depth']
    if not all(col in depth_profiles_df.columns for col in required_columns):
        raise ValueError(f"Required columns ('image_id', 'depth') not found in depth profiles file: {depth_profiles_path}")
    
    depth_profiles_df = depth_profiles_df[['image_id', 'depth']]

    return DetectionData(
        **metadata,
        input_path=input_path,
        output_path=output_path,
        depth_profiles_df=depth_profiles_df
    )

def get_detector_config() -> dict:
    """
    Returns a dictionary with the object detection parameters.
    """
    return {
        'threshold_value': THRESHOLD_VALUE,
        'threshold_max': THRESHOLD_MAX,
        'min_object_size': MIN_OBJECT_SIZE,
        'max_object_size': MAX_OBJECT_SIZE,
        'max_eccentricity': MAX_ECCENTRICITY,
        'max_mean_intensity': MAX_MEAN_INTENSITY,
        'min_major_axis_length': MIN_MAJOR_AXIS_LENGTH,
        'max_min_intensity': MAX_MIN_INTENSITY,
        'small_object_threshold': SMALL_OBJECT_THRESHOLD,
        'medium_object_threshold': MEDIUM_OBJECT_THRESHOLD,
        'large_object_padding': LARGE_OBJECT_PADDING,
        'small_object_padding': SMALL_OBJECT_PADDING,
        'medium_object_padding': MEDIUM_OBJECT_PADDING
    } 