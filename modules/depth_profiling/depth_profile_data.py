from dataclasses import dataclass
import argparse
from pathlib import Path
import os
from typing import Tuple, List
import datetime

from modules.common.parser import parse_metadata
from .utils import find_single_csv_file
from modules.common.fs_utils import ensure_dir

@dataclass(frozen=True)
class CsvParams:
    """Parameters for parsing CSV files."""
    separator: str
    header_row: int
    columns: Tuple[int, int]
    skipfooter: int
    extension: str
    time_column_name: str
    depth_column_name: str

@dataclass(frozen=True)
class DepthParams:
    """Parameters for depth calculation."""
    pressure_sensor_depth_multiplier: float
    image_height_cm: float
    image_height_pixels: int
    overlap_correction_depth_multiplier: float

@dataclass(frozen=True)
class RunMetadata:
    """Metadata for a specific run."""
    raw_img_paths: List[str]
    recording_start_date: datetime.date
    recording_start_time: datetime.time
    total_replicates: int
    project: str
    cycle: str
    location: str

@dataclass(frozen=True)
class DepthProfilingData:
    """Dataclass for depth profiling data."""
    run_metadata: RunMetadata
    pressure_sensor_csv_path: str
    output_path: str
    capture_rate: float
    csv_params: CsvParams
    depth_params: DepthParams

def process_arguments(
    args: argparse.Namespace, 
    csv_params: CsvParams, 
    depth_params: DepthParams
) -> DepthProfilingData:
    """
    Processes command line arguments and prepares data for depth profiling.

    Args:
        args (argparse.Namespace): The parsed command line arguments.
        csv_params (CsvParams): Parameters for CSV parsing.
        depth_params (DepthParams): Parameters for depth calculation.

    Returns:
        DepthProfilingData: A dataclass containing all necessary data for depth profiling.
    """
    if args.capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")

    input_path = Path(args.input)
    metadata_dict = parse_metadata(input_path)
    run_metadata = RunMetadata(**metadata_dict)
    
    pressure_sensor_csv_path = find_single_csv_file(input_path)

    output_dir = ensure_dir(args.output)
    date_str = run_metadata.recording_start_date.strftime("%Y%m%d")
    output_path = os.path.join(
        output_dir, 
        run_metadata.project, 
        date_str, 
        run_metadata.cycle, 
        run_metadata.location
    )
    os.makedirs(output_path, exist_ok=True)
    
    return DepthProfilingData(
        run_metadata=run_metadata,
        pressure_sensor_csv_path=pressure_sensor_csv_path,
        output_path=output_path,
        capture_rate=args.capture_rate,
        csv_params=csv_params,
        depth_params=depth_params
    )
