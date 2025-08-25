import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta
from typing import List

from .depth_profile_data import DepthProfilingData, CsvParams, DepthParams

def _load_pressure_sensor_csv(
    csv_path: str, 
    csv_params: CsvParams,
    depth_multiplier: float
) -> pd.DataFrame:
    """Load and process CSV depth data."""
    pressure_sensor_df = pd.read_csv(
        csv_path, 
        sep=csv_params.separator, 
        header=csv_params.header_row,
        usecols=list(csv_params.columns),
        names=[csv_params.time_column_name, csv_params.depth_column_name], 
        index_col=csv_params.time_column_name,
        skipfooter=csv_params.skipfooter, 
        engine='python'
    )
    
    # Process timestamps and depths
    pressure_sensor_df.index = pd.to_datetime(pressure_sensor_df.index, format='%d.%m.%Y %H:%M:%S.%f')
    pressure_sensor_df[csv_params.depth_column_name] = pressure_sensor_df[csv_params.depth_column_name].str.replace(',', '.').astype(float) * depth_multiplier
    
    return pressure_sensor_df

def _calculate_depths(
    pressure_sensor_df: pd.DataFrame, 
    image_paths: List[str], 
    recording_start_datetime: datetime, 
    capture_rate: float
) -> pd.Series:
    # Convert capture rate (in Hz) to a timedelta object
    timestep = relativedelta.relativedelta(microseconds=1/capture_rate*1000000)
    timestamps = [recording_start_datetime + (i * timestep) for i in range(len(image_paths))]
    nearest_indices = pressure_sensor_df.index.get_indexer(timestamps, method='nearest')
    return pressure_sensor_df.iloc[nearest_indices][pressure_sensor_df.columns[0]].values

def _calculate_pixel_overlap(
    depths: np.ndarray, 
    depth_params: DepthParams
) -> np.ndarray:
    """Calculate pixel overlaps for depth correction."""
    image_bottom_depths = depths * depth_params.overlap_correction_depth_multiplier + depth_params.image_height_cm
    image_top_depths = depths * depth_params.overlap_correction_depth_multiplier
    
    # Get the overlaps in cm
    overlaps_cm = np.zeros(len(depths))
    overlaps_cm[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
    
    # Convert the cm overlaps to pixels
    overlaps_pixels = np.round((overlaps_cm / depth_params.image_height_cm) * depth_params.image_height_pixels).astype(int)
    return overlaps_pixels

def _create_depth_dataframe(
    image_paths: List[str], 
    depth_values: pd.Series, 
    overlaps: np.ndarray, 
    depth_column_name: str
) -> pd.DataFrame:
    image_ids = [int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]) for p in image_paths]
    depth_mapping = {
        "image_path": [os.path.abspath(p) for p in image_paths],
        "image_id": image_ids,
        depth_column_name: depth_values,
        "pixel_overlap": overlaps
    }
    return pd.DataFrame(depth_mapping)

def profile_depths(data: DepthProfilingData):
    """
    Processes a group of images to calculate depth for each one.

    Args:
        data (DepthProfilingData): A dataclass containing all necessary data for depth profiling.
    """
    if not data.run_metadata.raw_img_paths:
        print("[PROFILING]: Warning: Empty image group provided.")
        return

    try:
        start_datetime = datetime.combine(data.run_metadata.recording_start_date, data.run_metadata.recording_start_time)
        
        pressure_sensor_df = _load_pressure_sensor_csv(
            data.pressure_sensor_csv_path,
            data.csv_params,
            data.depth_params.pressure_sensor_depth_multiplier
        )
        
        depth_values = _calculate_depths(
            pressure_sensor_df, 
            data.run_metadata.raw_img_paths, 
            start_datetime, 
            data.capture_rate
        )
        
        overlaps = _calculate_pixel_overlap(
            depth_values, 
            data.depth_params
        )
        
        mapped_df = _create_depth_dataframe(
            data.run_metadata.raw_img_paths, 
            depth_values, 
            overlaps, 
            data.csv_params.depth_column_name
        )
        
        output_csv_path = os.path.join(data.output_path, "depth_profiles" + data.csv_params.extension)
        mapped_df.to_csv(output_csv_path, index=False)
        print(f"[PROFILING]: Successfully saved data to {output_csv_path}")
        print(f"[PROFILING]: Processing completed successfully!")

    except (ValueError, FileNotFoundError) as e:
        print(f"[PROFILING]: Error processing CSV file: {e}")
        raise
    except Exception as e:
        print(f"[PROFILING]: An unexpected error occurred: {e}")
        raise
