import os
import pandas as pd
from datetime import datetime
from modules.common.constants import get_timestep_from_rate
from typing import List
from .config import ProfileConfig

class DepthProfiler:
    """Main class for depth profiling operations."""
    
    def __init__(self, config: ProfileConfig = None):
        self.config = config or ProfileConfig()
        self.timestep = get_timestep_from_rate(self.config.capture_rate)

    def _load_pressure_sensor_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and process CSV depth data."""
        pressure_sensor_df = pd.read_csv(
            csv_path, 
            sep=self.config.csv_separator, 
            header=self.config.csv_header_row,
            usecols=self.config.csv_columns,
            names=['time', 'depth'], 
            index_col='time',
            skipfooter=self.config.csv_skipfooter, 
            engine='python'
        )
        
        # Process timestamps and depths
        pressure_sensor_df.index = pd.to_datetime(pressure_sensor_df.index, format='%d.%m.%Y %H:%M:%S.%f')
        pressure_sensor_df['depth'] = pressure_sensor_df['depth'].str.replace(',', '.').astype(float) * self.config.depth_multiplier
        
        return pressure_sensor_df

    def _calculate_depths(self, pressure_sensor_df: pd.DataFrame, image_paths: List[str], recording_start_datetime: datetime) -> pd.Series:
        timestamps = [recording_start_datetime + (i * self.timestep) for i in range(len(image_paths))]
        nearest_indices = pressure_sensor_df.index.get_indexer(timestamps, method='nearest')
        return pressure_sensor_df.iloc[nearest_indices]['depth'].values

    def _create_depth_dataframe(self, image_paths: List[str], depth_values: pd.Series) -> pd.DataFrame:
        depth_mapping = {
            "image_path": [os.path.abspath(p) for p in image_paths],
            "depth": depth_values
        }
        return pd.DataFrame(depth_mapping)

    def process_depth_data(self, image_paths: List[str], pressure_sensor_csv_path: str, recording_start_datetime: datetime) -> pd.DataFrame:
        """
        Processes a group of images to calculate depth for each one.

        Args:
            image_paths (List[str]): A list of full paths to the images in the group.
            pressure_sensor_csv_path (str): The path to the associated CSV file.
            recording_start_datetime (datetime): The timestamp of the first image in the group.

        Returns:
            pd.DataFrame: A DataFrame with image paths and their corresponding depths, or None on failure.
        """
        if not image_paths:
            print("[PROFILING]: Warning: Empty image group provided.")
            return None

        print(f"[PROFILING]: Starting depth matching for a group of {len(image_paths)} images...")

        try:
            pressure_sensor_df = self._load_pressure_sensor_csv(pressure_sensor_csv_path)
            depth_values = self._calculate_depths(pressure_sensor_df, image_paths, recording_start_datetime)
            mapped_df = self._create_depth_dataframe(image_paths, depth_values)
            return mapped_df
        except (ValueError, FileNotFoundError) as e:
            print(f"[PROFILING]: Error processing CSV file: {e}")
            return None
        except Exception as e:
            print(f"[PROFILING]: An unexpected error occurred: {e}")
            return None
