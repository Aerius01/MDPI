import os
from pathlib import Path
from typing import Union, List
import pandas as pd
from modules.common.parser import parse_metadata, find_single_csv_file

class Profile:
    """A class to hold profile information for a pipeline run."""

    def __init__(self, directory_path: Union[str, os.PathLike], output_folder: Union[str, os.PathLike]):
        """
        Initializes the Profile object by parsing a directory path and an image filename within it.

        Args:
            directory_path: Path to the directory containing the images.
            output_folder: Path to the directory where output files will be saved.
        """
        # Passed arguments  
        self.directory_path = Path(directory_path).resolve()
        self.output_folder = Path(output_folder).resolve()
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.pressure_sensor_csv_path = find_single_csv_file(self.directory_path)
        
        # Parse metadata
        metadata = parse_metadata(self.directory_path)
        self.project = metadata["project"]
        self.recording_start_date = metadata["recording_start_date"]
        self.cycle = metadata["cycle"]
        self.location = metadata["location"]
        self.total_replicates = metadata["total_replicates"]
        self.recording_start_time = metadata["recording_start_time"]
        self.raw_img_paths = metadata["raw_img_paths"]
        
        # Create the output path
        self.output_path = os.path.join(self.output_folder, self.project, self.recording_start_date.strftime("%Y%m%d"), self.cycle, self.location)
        os.makedirs(self.output_path, exist_ok=True)

        # Processed pipeline data null initializations
        self.deduplicated_imgs: List[str] = []
        self.depth_mapping_df: pd.DataFrame = None

    def set_deduplicated_imgs(self, removed_paths: List[str]):
        """
        Sets the deduplicated_imgs attribute by removing the specified paths from raw_img_paths.

        Args:
            removed_paths: A list of file paths that have been removed.
        """
        remaining_files = sorted(list(set(self.raw_img_paths) - set(removed_paths)))
        self.deduplicated_imgs = remaining_files 

    def set_depth_mapping_df(self, depth_df: pd.DataFrame):
        """
        Sets the depth DataFrame for the profile.

        Args:
            depth_df: A pandas DataFrame containing depth information.
        """
        self.depth_mapping_df = depth_df 