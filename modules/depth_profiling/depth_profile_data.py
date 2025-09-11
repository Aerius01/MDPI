from dataclasses import dataclass
import argparse
from pathlib import Path
import os
from typing import Tuple, List
import datetime
import pandas as pd
import cv2

from modules.common.constants import CONSTANTS
from modules.common.parser import parse_metadata
from .utils import find_single_csv_file
from modules.common.fs_utils import ensure_dir

# ─── M D P I · P A R A M E T E R S ──────────────────────────────────────────────────
# The MDPI is placed horizontally as it moves through the water column,
# meaning that images are vertical. 4.3 cm is the height of the field of view.
# This is important for the calculation of the pixel overlap.
IMAGE_HEIGHT_CM: float = 4.3

# Old camera format defaults
OLD_FORMAT_HEADER_ROW: int = 4   # Row index of the header (0-based)
OLD_FORMAT_SKIPFOOTER: int = 1   # How many rows to skip after the footer?
OLD_FORMAT_TIME_COLUMN_SEARCH: str = "Time SN:"  # Old format time column search term
OLD_FORMAT_DEPTH_COLUMN_SEARCH: str = "bar"  # Old format depth column search term (pressure in bar)
OLD_FORMAT_PRESSURE_MULTIPLIER: float = 10.0  # Convert pressure sensor values to depth (meters)

# New camera format constants
NEW_FORMAT_HEADER_ROW: int = 0  # New format has single header line
NEW_FORMAT_SKIPFOOTER: int = 0  # New format has no footer to skip
NEW_FORMAT_TIME_COLUMN_SEARCH: str = "Date-Time"  # New format time column name
NEW_FORMAT_DEPTH_COLUMN_SEARCH: str = "Depth(m)"  # New format depth column name
NEW_FORMAT_PRESSURE_MULTIPLIER: float = 1.0  # New format already in meters

PRESSURE_SENSOR_CSV_SEPARATOR = ';' # Separator to decode the pressure sensor CSV file


@dataclass(frozen=True)
class CsvParams:
    """Parameters for parsing CSV files."""
    separator: str
    header_row: int
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
    camera_format: str

def read_csv_with_encodings(csv_path: str, **kwargs) -> pd.DataFrame:
    """Reads a CSV file by trying multiple common encodings."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, SyntaxError):
            continue
    raise ValueError(f"Could not read file {csv_path} with any of the encodings: {encodings}")

def detect_camera_format(df: pd.DataFrame) -> str:
    """Detects the camera format based on the first column name of the CSV."""
    if df.columns.empty:
        raise ValueError("Input DataFrame has no columns.")

    first_column_name = df.columns[0]
    if first_column_name == "Name":
        return "old"
    if first_column_name == "Number":
        return "new"

    raise ValueError(
        f"Could not determine camera format. "
        f"First column name is '{first_column_name}', but expected 'Name' or 'Number'."
    )

def create_camera_parameters(is_new_format: bool, image_height_pixels: int) -> Tuple[CsvParams, DepthParams]:
    """Create appropriate CSV and depth parameters based on detected camera format."""
    csv_params = CsvParams(
        separator=PRESSURE_SENSOR_CSV_SEPARATOR,
        header_row=NEW_FORMAT_HEADER_ROW if is_new_format else OLD_FORMAT_HEADER_ROW,
        skipfooter=NEW_FORMAT_SKIPFOOTER if is_new_format else OLD_FORMAT_SKIPFOOTER,
        extension=CONSTANTS.CSV_EXTENSION,
        time_column_name=NEW_FORMAT_TIME_COLUMN_SEARCH if is_new_format else OLD_FORMAT_TIME_COLUMN_SEARCH,
        depth_column_name=NEW_FORMAT_DEPTH_COLUMN_SEARCH if is_new_format else OLD_FORMAT_DEPTH_COLUMN_SEARCH
    )

    depth_params = DepthParams(
        pressure_sensor_depth_multiplier=NEW_FORMAT_PRESSURE_MULTIPLIER if is_new_format else OLD_FORMAT_PRESSURE_MULTIPLIER,
        image_height_cm=IMAGE_HEIGHT_CM,
        image_height_pixels=image_height_pixels
    )
    
    return csv_params, depth_params

def process_arguments(args: argparse.Namespace) -> DepthProfilingData:
    """
    Processes command line arguments and prepares data for depth profiling.
    Automatically detects camera format and configures all parameters accordingly.
    """
    if args.capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")

    input_path = Path(args.input)
    metadata_dict = parse_metadata(input_path)
    run_metadata = RunMetadata(**metadata_dict)
    
    # Read the first image to determine the image height in pixels
    if not run_metadata.raw_img_paths:
        raise FileNotFoundError("No image files found in the input directory.")
        
    first_image_path = run_metadata.raw_img_paths[0]
    image = cv2.imread(first_image_path)
    if image is None:
        raise ValueError(f"Could not read image file: {first_image_path}")
    image_height_pixels = image.shape[0]

    pressure_sensor_csv_path = find_single_csv_file(input_path)
    
    header_df = read_csv_with_encodings(
        pressure_sensor_csv_path, 
        sep=PRESSURE_SENSOR_CSV_SEPARATOR, 
        header=0, 
        engine='python', 
        nrows=0
    )
    
    camera_format = detect_camera_format(header_df)
    print(f"[PROFILING]: Detected {camera_format} camera format.")
    
    final_csv_params, final_depth_params = create_camera_parameters(camera_format == "new", image_height_pixels)

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
        csv_params=final_csv_params,
        depth_params=final_depth_params,
        camera_format=camera_format
    )
