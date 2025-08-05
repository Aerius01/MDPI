import os
import pandas as pd
from datetime import datetime
from dateutil import relativedelta
from modules.common.constants import CONSTANTS
from typing import List

def get_timestep_from_rate(capture_rate: float) -> relativedelta.relativedelta:
    """Calculate timestep from capture rate in Hz."""
    if capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")
    return relativedelta.relativedelta(microseconds=1/capture_rate*1000000)

class DepthProfiler:
    """Main class for depth profiling operations."""
    
    def __init__(self):
        pass

    def _load_pressure_sensor_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and process CSV depth data."""
        pressure_sensor_df = pd.read_csv(
            csv_path, 
            sep=CONSTANTS.CSV_SEPARATOR, 
            header=CONSTANTS.CSV_HEADER_ROW,
            usecols=list(CONSTANTS.CSV_COLUMNS),
            names=['time', 'depth'], 
            index_col='time',
            skipfooter=CONSTANTS.CSV_SKIPFOOTER, 
            engine='python'
        )
        
        # Process timestamps and depths
        pressure_sensor_df.index = pd.to_datetime(pressure_sensor_df.index, format='%d.%m.%Y %H:%M:%S.%f')
        pressure_sensor_df['depth'] = pressure_sensor_df['depth'].str.replace(',', '.').astype(float) * CONSTANTS.DEPTH_MULTIPLIER
        
        return pressure_sensor_df

    def _calculate_depths(self, pressure_sensor_df: pd.DataFrame, image_paths: List[str], recording_start_datetime: datetime, capture_rate: float) -> pd.Series:
        timestep = get_timestep_from_rate(capture_rate)
        timestamps = [recording_start_datetime + (i * timestep) for i in range(len(image_paths))]
        nearest_indices = pressure_sensor_df.index.get_indexer(timestamps, method='nearest')
        return pressure_sensor_df.iloc[nearest_indices]['depth'].values

    def _create_depth_dataframe(self, image_paths: List[str], depth_values: pd.Series) -> pd.DataFrame:
        depth_mapping = {
            "image_path": [os.path.abspath(p) for p in image_paths],
            "depth": depth_values
        }
        return pd.DataFrame(depth_mapping)

    def map_images_to_depths(self, image_paths: List[str], pressure_sensor_csv_path: str, recording_start_datetime: datetime, capture_rate: float) -> pd.DataFrame:
        """
        Processes a group of images to calculate depth for each one.

        Args:
            image_paths (List[str]): A list of full paths to the images in the group.
            pressure_sensor_csv_path (str): The path to the associated CSV file.
            recording_start_datetime (datetime): The timestamp of the first image in the group.
            capture_rate (float): The capture rate in Hz.

        Returns:
            pd.DataFrame: A DataFrame with image paths and their corresponding depths, or None on failure.
        """
        if not image_paths:
            print("[PROFILING]: Warning: Empty image group provided.")
            return None

        try:
            pressure_sensor_df = self._load_pressure_sensor_csv(pressure_sensor_csv_path)
            depth_values = self._calculate_depths(pressure_sensor_df, image_paths, recording_start_datetime, capture_rate)
            mapped_df = self._create_depth_dataframe(image_paths, depth_values)
            return mapped_df
        except (ValueError, FileNotFoundError) as e:
            print(f"[PROFILING]: Error processing CSV file: {e}")
            return None
        except Exception as e:
            print(f"[PROFILING]: An unexpected error occurred: {e}")
            return None
