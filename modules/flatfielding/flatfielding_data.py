from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import os
from modules.common.constants import CONSTANTS
from types import SimpleNamespace

# ─── F L A T F I E L D I N G · P A R A M E T E R S ──────────────────────────────────
BATCH_SIZE = CONSTANTS.BATCH_SIZE # How many images to process at a given time
NORMALIZATION_FACTOR = 235 # Empirical flatfielding factor → set by Tim W. & Jens N.
IMAGE_SAVING_EXTENSION = '.jpeg' # Extension to use when saving the flatfielded images

@dataclass
class FlatfieldingData:
    output_path: str
    metadata: dict
    overlap_map: dict
    batch_size: int
    normalization_factor: int
    image_extension: str

def validate_depth_profiles(depth_df: pd.DataFrame):
    """
    Validates the structure and content of the depth profiles DataFrame.
    Raises ValueError if any checks fail.
    """
    # Check for required columns
    if 'pixel_overlap' not in depth_df.columns:
        raise ValueError("Depth profiles CSV file must contain a 'pixel_overlap' column.")
    if 'image_path' not in depth_df.columns:
        raise ValueError("Depth profiles CSV file must contain an 'image_path' column.")

    # Check for empty data in columns
    if depth_df['pixel_overlap'].isnull().any():
        raise ValueError("The 'pixel_overlap' column in the depth profiles CSV file contains missing values.")
    if depth_df['image_path'].isnull().any():
        raise ValueError("The 'image_path' column in the depth profiles CSV file contains missing values.")

    # Check for duplicate image paths
    if depth_df['image_path'].duplicated().any():
        duplicates = depth_df[depth_df['image_path'].duplicated()]['image_path'].tolist()
        raise ValueError(f"The 'image_path' column in the depth profiles CSV file contains duplicate values: {duplicates}")

def process_arguments(run_config: SimpleNamespace, depth_df: pd.DataFrame) -> FlatfieldingData:
    # Create a mapping from absolute image path to pixel_overlap value
    validate_depth_profiles(depth_df)
    overlap_map = pd.Series(depth_df.pixel_overlap.values, index=depth_df.image_path).to_dict()

    # Create output path
    output_path = os.path.join(Path(run_config.output_root), CONSTANTS.FLATFIELD_DIR_NAME)
    os.makedirs(output_path, exist_ok=True)

    return FlatfieldingData(
        output_path=output_path,
        metadata=run_config.metadata,
        overlap_map=overlap_map,
        batch_size=BATCH_SIZE,
        normalization_factor=NORMALIZATION_FACTOR,
        image_extension=IMAGE_SAVING_EXTENSION
    )