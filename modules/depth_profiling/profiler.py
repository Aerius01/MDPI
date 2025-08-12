import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta
from typing import List, Tuple
from modules.common import CONSTANTS

# Destructured CONSTANTS for cleaner readability
IMAGE_HEIGHT_CM = CONSTANTS.IMAGE_HEIGHT_CM
IMAGE_HEIGHT_PIXELS = CONSTANTS.IMAGE_HEIGHT_PIXELS
PRESSURE_SENSOR_DEPTH_MULTIPLIER = CONSTANTS.PRESSURE_SENSOR_DEPTH_MULTIPLIER
OVERLAP_CORRECTION_DEPTH_MULTIPLIER = CONSTANTS.OVERLAP_CORRECTION_DEPTH_MULTIPLIER

class DepthProfiler:
    """Main class for depth profiling operations."""
    
    def __init__(self,
                 csv_separator: str,
                 csv_header_row: int,
                 csv_columns: Tuple,
                 csv_skipfooter: int,
                 depth_multiplier: float,
                 time_column_name: str,
                 depth_column_name: str):
        self.csv_separator = csv_separator
        self.csv_header_row = csv_header_row
        self.csv_columns = csv_columns
        self.csv_skipfooter = csv_skipfooter
        self.depth_multiplier = depth_multiplier
        self.time_column_name = time_column_name
        self.depth_column_name = depth_column_name

    def _load_pressure_sensor_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and process CSV depth data."""
        pressure_sensor_df = pd.read_csv(
            csv_path, 
            sep=self.csv_separator, 
            header=self.csv_header_row,
            usecols=list(self.csv_columns),
            names=[self.time_column_name, self.depth_column_name], 
            index_col=self.time_column_name,
            skipfooter=self.csv_skipfooter, 
            engine='python'
        )
        
        # Process timestamps and depths
        pressure_sensor_df.index = pd.to_datetime(pressure_sensor_df.index, format='%d.%m.%Y %H:%M:%S.%f')
        pressure_sensor_df[self.depth_column_name] = pressure_sensor_df[self.depth_column_name].str.replace(',', '.').astype(float) * self.depth_multiplier
        
        return pressure_sensor_df

    def _calculate_depths(self, pressure_sensor_df: pd.DataFrame, image_paths: List[str], recording_start_datetime: datetime, capture_rate: float) -> pd.Series:
        # Convert capture rate (in Hz) to a timedelta object
        timestep = relativedelta.relativedelta(microseconds=1/capture_rate*1000000)
        timestamps = [recording_start_datetime + (i * timestep) for i in range(len(image_paths))]
        nearest_indices = pressure_sensor_df.index.get_indexer(timestamps, method='nearest')
        return pressure_sensor_df.iloc[nearest_indices][self.depth_column_name].values

    def _calculate_pixel_overlap(self, depths: np.ndarray) -> np.ndarray:
        """Calculate pixel overlaps for depth correction."""
       
        image_bottom_depths = depths * OVERLAP_CORRECTION_DEPTH_MULTIPLIER + IMAGE_HEIGHT_CM
        image_top_depths = depths * OVERLAP_CORRECTION_DEPTH_MULTIPLIER
        
        # Get the overlaps in cm
        overlaps_cm = np.zeros(len(depths)) # Initializing the array
        overlaps_cm[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
        
        # Convert the cm overlaps to pixels
        overlaps_pixels = np.round((overlaps_cm / IMAGE_HEIGHT_CM) * IMAGE_HEIGHT_PIXELS).astype(int)
        return overlaps_pixels

    def _create_depth_dataframe(self, image_paths: List[str], depth_values: pd.Series, overlaps: np.ndarray) -> pd.DataFrame:
        depth_mapping = {
            "image_path": [os.path.abspath(p) for p in image_paths],
            self.depth_column_name: depth_values,
            "pixel_overlap": overlaps
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
            overlaps = self._calculate_pixel_overlap(depth_values)
            mapped_df = self._create_depth_dataframe(image_paths, depth_values, overlaps)
            return mapped_df
        except (ValueError, FileNotFoundError) as e:
            print(f"[PROFILING]: Error processing CSV file: {e}")
            return None
        except Exception as e:
            print(f"[PROFILING]: An unexpected error occurred: {e}")
            return None
