import datetime
import os
from pathlib import Path
from typing import Union, List
import pandas as pd
from modules.common.parser import parse_path_metadata, parse_file_metadata, find_single_csv_file

class Profile:
    """A class to hold profile information for a pipeline run."""

    def __init__(self, directory_path: Union[str, os.PathLike], output_folder: Union[str, os.PathLike]):
        """
        Initializes the Profile object by parsing a directory path and an image filename within it.

        Args:
            directory_path: Path to the directory containing the images.
            output_folder: Path to the directory where output files will be saved.
        """
        self.directory_path = Path(directory_path).resolve()
        self.output_folder = Path(output_folder).resolve()
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.deduplicated_imgs: List[str] = []
        self.depth_mapping_df: pd.DataFrame = None
        
        path_metadata = parse_path_metadata(self.directory_path)
        self.project = path_metadata["project"]
        self.date_str = path_metadata["date_str"]
        self.cycle = path_metadata["cycle"]
        self.location = path_metadata["location"]
        self.recording_start_date = path_metadata["recording_start_date"]

        # Parse file metadata from directory path and recording date
        file_metadata = parse_file_metadata(self.directory_path, self.recording_start_date)
        self.replicate = file_metadata["replicate"]
        self.recording_start_time = file_metadata["recording_start_time"]
        self.raw_imgs = file_metadata["raw_imgs"]
        
        self.pressure_sensor_csv_path = find_single_csv_file(self.directory_path)

    def set_deduplicated_imgs(self, removed_paths: List[str]):
        """
        Sets the deduplicated_imgs attribute by removing the specified paths from raw_imgs.

        Args:
            removed_paths: A list of file paths that have been removed.
        """
        remaining_files = sorted(list(set(self.raw_imgs) - set(removed_paths)))
        self.deduplicated_imgs = remaining_files 

    def set_depth_mapping_df(self, depth_df: pd.DataFrame):
        """
        Sets the depth DataFrame for the profile.

        Args:
            depth_df: A pandas DataFrame containing depth information.
        """
        self.depth_mapping_df = depth_df 