import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import relativedelta
from typing import List
from pathlib import Path

from .depth_profile_data import DepthProfilingData, CsvParams, DepthParams, read_csv_with_encodings

def _load_pressure_sensor_csv(
    csv_path: str, 
    csv_params: CsvParams,
    depth_multiplier: float,
    camera_format: str
) -> pd.DataFrame:
    """Load and process CSV depth data using pandas chaining."""
    df = read_csv_with_encodings(
        csv_path,
        sep=csv_params.separator,
        header=csv_params.header_row,
        skipfooter=csv_params.skipfooter,
        engine='python'
    )

    time_col = next((col for col in df.columns if csv_params.time_column_name.lower() in col.lower()), None)
    depth_col = next((col for col in df.columns if csv_params.depth_column_name.lower() in col.lower()), None)

    if not time_col:
        raise ValueError(f"Time column containing '{csv_params.time_column_name}' not found.")
    if not depth_col:
        raise ValueError(f"Depth column containing '{csv_params.depth_column_name}' not found.")

    # Define the correct time format based on camera type
    time_format = '%d.%m.%Y %H:%M:%S,%f' if camera_format == "new" else '%d.%m.%Y %H:%M:%S.%f'

    # Process the DataFrame using a clear, chained sequence of operations
    pressure_sensor_df = (
        df[[time_col, depth_col]]
        .rename(columns={time_col: "time", depth_col: "depth"})
        .assign(
            time=lambda x: pd.to_datetime(x["time"], format=time_format),
            depth=lambda x: pd.to_numeric(x["depth"].astype(str).str.replace(',', '.'), errors='coerce') * depth_multiplier
        )
        .set_index("time")
    )
    
    return pressure_sensor_df

def _calculate_depths(
    pressure_sensor_df: pd.DataFrame, 
    image_paths: List[str], 
    recording_start_datetime: datetime, 
    capture_rate: float
) -> pd.Series:
    """Calculate the depth for each image based on its timestamp."""
    timestep = relativedelta.relatedelta(microseconds=1_000_000 / capture_rate)
    timestamps = [recording_start_datetime + (i * timestep) for i in range(len(image_paths))]
    
    # Find the nearest depth measurement for each image timestamp
    nearest_indices = pressure_sensor_df.index.get_indexer(timestamps, method='nearest')
    return pressure_sensor_df.iloc[nearest_indices]['depth'].values

def _calculate_pixel_overlap(
    depths: np.ndarray, 
    depth_params: DepthParams
) -> np.ndarray:
    """Calculate pixel overlaps for depth correction."""
    # Calculate the depth of the top and bottom of each image in cm
    image_top_depths_cm = depths * depth_params.overlap_correction_depth_multiplier
    image_bottom_depths_cm = image_top_depths_cm + depth_params.image_height_cm
    
    # Calculate overlaps in cm, ensuring no negative values
    overlaps_cm = np.zeros_like(depths, dtype=float)
    overlaps_cm[1:] = np.maximum(0, image_bottom_depths_cm[:-1] - image_top_depths_cm[1:])
    
    # Convert overlaps to pixels
    return np.round(
        (overlaps_cm / depth_params.image_height_cm) * depth_params.image_height_pixels
    ).astype(int)

def _create_depth_dataframe(
    image_paths: List[str], 
    depth_values: pd.Series, 
    overlaps: np.ndarray
) -> pd.DataFrame:
    """Create a DataFrame to store depth information for each image."""
    image_ids = [int(Path(p).stem.split('_')[-1]) for p in image_paths]
    depth_mapping = {
        "image_path": [os.path.abspath(p) for p in image_paths],
        "image_id": image_ids,
        "depth": depth_values,
        "pixel_overlap": overlaps
    }
    return pd.DataFrame(depth_mapping)

def profile_depths(data: DepthProfilingData):
    """
    Processes a group of images to calculate depth for each one.
    """
    if not data.run_metadata.raw_img_paths:
        print("[PROFILING]: Warning: Empty image group provided.")
        return

    try:
        start_datetime = datetime.combine(data.run_metadata.recording_start_date, data.run_metadata.recording_start_time)
        
        pressure_sensor_df = _load_pressure_sensor_csv(
            data.pressure_sensor_csv_path,
            data.csv_params,
            data.depth_params.pressure_sensor_depth_multiplier,
            data.camera_format
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
            overlaps
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
