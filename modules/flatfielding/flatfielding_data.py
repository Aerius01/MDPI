from dataclasses import dataclass
import argparse
from pathlib import Path
import pandas as pd
import os
from modules.common.parser import parse_metadata
from modules.common.constants import CONSTANTS
from modules.common.cli_utils import CommonCLI

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE = CONSTANTS.BATCH_SIZE
NORMALIZATION_FACTOR = CONSTANTS.NORMALIZATION_FACTOR
IMAGE_EXTENSION = CONSTANTS.JPEG_EXTENSION

@dataclass
class FlatfieldingData:
    output_path: str
    metadata: dict
    overlap_map: dict
    batch_size: int
    normalization_factor: int
    image_extension: str

def process_arguments(args: argparse.Namespace) -> FlatfieldingData:
    # Get image paths from input directory
    print(f"[FLATFIELDING]: Loading images from {args.input}")
    input_path = Path(args.input)
    metadata = parse_metadata(input_path)

    # Load depth profiles to get pixel overlaps
    print(f"[FLATFIELDING]: Loading depth profiles from {args.depth_profiles}")
    depth_df = pd.read_csv(args.depth_profiles)
    # Create a mapping from absolute image path to pixel_overlap value
    overlap_map = pd.Series(depth_df.pixel_overlap.values, index=depth_df.image_path).to_dict()

    # Validate and create output path
    output_dir = CommonCLI.validate_output_path(args.output)
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(output_dir, metadata["project"], date_str, metadata["cycle"], metadata["location"], 
                              "flatfielded_images")
    os.makedirs(output_path, exist_ok=True)

    return FlatfieldingData(
        output_path=output_path,
        metadata=metadata,
        overlap_map=overlap_map,
        batch_size=BATCH_SIZE,
        normalization_factor=NORMALIZATION_FACTOR,
        image_extension=IMAGE_EXTENSION
    )