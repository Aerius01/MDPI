import argparse
from pathlib import Path
import pandas as pd
from modules.common.cli_utils import CommonCLI
from modules.common.parser import parse_flatfield_metadata_from_directory
import os
from dataclasses import dataclass, fields

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

    output_dir = CommonCLI.validate_output_path(args.output)
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