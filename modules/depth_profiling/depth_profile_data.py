from dataclasses import dataclass
import argparse
from pathlib import Path
import os
from typing import Tuple, List
import datetime
import pandas as pd

from modules.common.parser import parse_metadata
from .utils import find_single_csv_file
from modules.common.fs_utils import ensure_dir

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

def detect_camera_format(csv_path: str) -> Tuple[bool, str]:
    """
    Detect whether CSV file is from old or new camera format.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (is_new_format, detected_format_description)
    """
    try:
        # Try different encodings for reading the file
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        lines = []
        
        for encoding in encodings:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    # Read first 8 lines to detect format
                    lines = []
                    for _ in range(8):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line.strip())
                break  # If successful, exit the encoding loop
            except UnicodeDecodeError:
                continue
        
        if not lines:
            return False, "Could not read file with any encoding, defaulting to old camera format"
        
        # Check if first line contains 'Date-Time' and 'Depth(m)' (new format)
        first_line = lines[0] if lines else ""
        if 'Date-Time' in first_line and 'Depth(m)' in first_line:
            return True, "New camera format (single header line, depths in meters)"
            
        # Check for old format characteristics (complex header structure)
        # Old format has a header on line 6 (index 5) with 'Time SN:' and 'bar'
        if len(lines) >= 6:
            header_line = lines[5] if len(lines) > 5 else ""
            if 'Time SN:' in header_line and 'bar' in header_line:
                return False, "Old camera format (complex header, depths in bar)"
        
        # Additional check for old format - look for the characteristic structure
        if len(lines) >= 2:
            # Check if there are empty lines and structured metadata (typical of old format)
            if any('Measure-Intervall' in line for line in lines[:5]):
                return False, "Old camera format (detected measure interval metadata)"
        
        # Fallback: if we can't clearly identify, assume old format for backward compatibility
        return False, "Unknown format, defaulting to old camera format"
        
    except Exception as e:
        print(f"[FORMAT DETECTION]: Warning - Could not detect format: {e}")
        return False, "Error detecting format, defaulting to old camera format"

def find_column_indices(csv_path: str, header_row: int, separator: str, time_column_name: str, depth_column_name: str) -> Tuple[int, int]:
    """
    Find column indices by searching for column names in the header.
    
    Args:
        csv_path: Path to CSV file
        header_row: Which row contains the header
        separator: CSV separator character
        time_column_name: Name of time column to find
        depth_column_name: Name of depth column to find
        
    Returns:
        Tuple of (time_column_index, depth_column_index)
        
    Raises:
        ValueError: If columns cannot be found
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        header_line = None
        
        for encoding in encodings:
            try:
                with open(csv_path, 'r', encoding=encoding) as f:
                    # Read lines until we reach the header row (0-based indexing)
                    lines = f.readlines()
                    if header_row < len(lines):
                        header_line = lines[header_row].strip()
                    else:
                        raise ValueError(f"Header row {header_row} does not exist in file")
                break
            except UnicodeDecodeError:
                continue
        
        if not header_line:
            raise ValueError(f"Could not read header from {csv_path}")
        
        # Split header into columns
        columns = header_line.split(separator)
        
        # Find time column (exact match or partial match)
        time_idx = None
        for i, col in enumerate(columns):
            if time_column_name in col:
                time_idx = i
                break
        
        # Find depth column (exact match or partial match)  
        depth_idx = None
        for i, col in enumerate(columns):
            if depth_column_name in col:
                depth_idx = i
                break
        
        if time_idx is None:
            raise ValueError(f"Could not find time column containing '{time_column_name}' in header: {columns}")
        if depth_idx is None:
            raise ValueError(f"Could not find depth column containing '{depth_column_name}' in header: {columns}")
            
        return time_idx, depth_idx
        
    except Exception as e:
        raise ValueError(f"Error finding columns in {csv_path}: {e}")

def create_camera_parameters(is_new_format: bool, base_csv_params: CsvParams, base_depth_params: DepthParams) -> Tuple[CsvParams, DepthParams]:
    """
    Create appropriate CSV and depth parameters based on detected camera format.
    
    Args:
        is_new_format: Whether this is the new camera format
        base_csv_params: Base CSV parameters (for old format)
        base_depth_params: Base depth parameters (for old format)
        
    Returns:
        Tuple of (csv_params, depth_params) adjusted for the detected format
    """
    from modules.common.constants import CONSTANTS
    
    if is_new_format:
        # New camera format parameters using constants
        csv_params = CsvParams(
            separator=base_csv_params.separator,  # Both formats use semicolon
            header_row=CONSTANTS.NEW_FORMAT_HEADER_ROW,
            skipfooter=CONSTANTS.NEW_FORMAT_SKIPFOOTER,
            extension=base_csv_params.extension,
            time_column_name=CONSTANTS.NEW_FORMAT_TIME_COLUMN_NAME,
            depth_column_name=CONSTANTS.NEW_FORMAT_DEPTH_COLUMN_NAME
        )
        
        # Depth parameters - new format already in meters
        depth_params = DepthParams(
            pressure_sensor_depth_multiplier=CONSTANTS.NEW_FORMAT_PRESSURE_MULTIPLIER,
            image_height_cm=base_depth_params.image_height_cm,
            image_height_pixels=base_depth_params.image_height_pixels,
            overlap_correction_depth_multiplier=base_depth_params.overlap_correction_depth_multiplier
        )
    else:
        # Use original parameters for old format
        csv_params = base_csv_params
        depth_params = base_depth_params
    
    return csv_params, depth_params

def process_arguments(args: argparse.Namespace) -> DepthProfilingData:
    """
    Processes command line arguments and prepares data for depth profiling.
    Automatically detects camera format and configures all parameters accordingly.

    Args:
        args (argparse.Namespace): The parsed command line arguments.

    Returns:
        DepthProfilingData: A dataclass containing all necessary data for depth profiling.
    """
    if args.capture_rate <= 0:
        raise ValueError("Capture rate must be a positive number.")

    input_path = Path(args.input)
    metadata_dict = parse_metadata(input_path)
    run_metadata = RunMetadata(**metadata_dict)
    
    pressure_sensor_csv_path = find_single_csv_file(input_path)
    
    # Detect camera format and adjust parameters
    is_new_format, format_description = detect_camera_format(pressure_sensor_csv_path)
    print(f"[FORMAT DETECTION]: {format_description}")
    
    # Import constants
    from modules.common.constants import CONSTANTS
    
    # Create base parameters using CONSTANTS (old format defaults for fallback)
    base_csv_params = CsvParams(
        separator=CONSTANTS.CSV_SEPARATOR,
        header_row=CONSTANTS.CSV_HEADER_ROW,
        skipfooter=CONSTANTS.CSV_SKIPFOOTER,
        extension=CONSTANTS.CSV_EXTENSION,
        time_column_name=CONSTANTS.OLD_FORMAT_TIME_COLUMN_SEARCH,
        depth_column_name=CONSTANTS.OLD_FORMAT_DEPTH_COLUMN_SEARCH
    )
    
    base_depth_params = DepthParams(
        pressure_sensor_depth_multiplier=CONSTANTS.PRESSURE_SENSOR_DEPTH_MULTIPLIER,
        image_height_cm=CONSTANTS.IMAGE_HEIGHT_CM,
        image_height_pixels=CONSTANTS.IMAGE_HEIGHT_PIXELS,
        overlap_correction_depth_multiplier=CONSTANTS.OVERLAP_CORRECTION_DEPTH_MULTIPLIER
    )
    
    # Get appropriate parameters for detected format
    final_csv_params, final_depth_params = create_camera_parameters(
        is_new_format, base_csv_params, base_depth_params
    )

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
        depth_params=final_depth_params
    )
