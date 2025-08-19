from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import pandas as pd
from modules.common.parser import parse_vignette_metadata
from modules.common.constants import CONSTANTS
from typing import Callable, Optional
from imutils import paths

# Destructure constants
DEFAULT_CATEGORIES = CONSTANTS.CLASSIFICATION_CATEGORIES

# Hardcoded constants
CSV_FILENAME = 'object_data.csv'
MODEL_CHECKPOINT_FILENAME = 'model.ckpt'
MODEL_CHECKPOINT_EXTENSIONS = ['meta', 'index', 'data-00000-of-00001']

@dataclass
class ValidatedArguments:
    vignette_paths: list
    output_path: str
    model_path: str
    metadata: dict
    batch_size: int
    input_size: int
    input_depth: int
    categories: list
    csv_filename: str
    detection_df: 'pd.DataFrame'

def _validate_model(model_path: str):
    # Ensure the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    
    # Ensure the required model checkpoint files exist
    model_checkpoint = os.path.join(model_path, MODEL_CHECKPOINT_FILENAME)
    if not any(os.path.exists(f"{model_checkpoint}.{ext}") for ext in MODEL_CHECKPOINT_EXTENSIONS):
        raise FileNotFoundError(f"Model checkpoint files not found in: {model_path}")
    
def _validate_input(vignettes_folder: str, sort_key: Optional[Callable] = None) -> list[str]:
    # Ensure the vignettes folder exists
    if not os.path.exists(vignettes_folder):
        raise FileNotFoundError(f"Vignettes folder does not exist: {vignettes_folder}")
    
    try:
        # Get a list of all vignette image paths and ensure it's not empty
        vignette_paths = list(paths.list_images(vignettes_folder))
        if not vignette_paths:
            raise ValueError(f"No vignettes found in specified folder: {vignettes_folder}")
        
        # Sort the vignette paths if a sort key is provided
        if sort_key:
            vignette_paths = sorted(vignette_paths, key=sort_key)
        else:
            vignette_paths = sorted(vignette_paths)
            
        return vignette_paths
    except Exception as e:
        raise ValueError(f"Error reading vignettes from {vignettes_folder}: {str(e)}")
    
def _validate_output_path(output_path: str) -> Path:
    try:
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        raise ValueError(f"Cannot create output path {output_path}: {str(e)}") 
    
def _validate_detection_csv(csv_path: str) -> pd.DataFrame:
    # Ensure the detection CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Detection CSV file not found at {csv_path}. It must be created by the object detection module first.")
    
    try:
        detection_df = pd.read_csv(csv_path, sep=None, engine='python')

        # Ensure the detection CSV file is not empty    
        if detection_df.empty:
            raise ValueError(f"Detection CSV file is empty: {csv_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Detection CSV file is empty: {csv_path}") from None
    
    return detection_df

def _get_vignette_sort_key(vignette_path: str) -> tuple[int, int]:
    """
    Sort key for vignettes. Sorts by replicate, then by object ID.
    The expected filename format is: {recording_start_time}_{replicate}_vignette_{object_id}.{ext}
    """
    filename = os.path.basename(vignette_path)
    base_filename = os.path.splitext(filename)[0]
    parts = base_filename.split('_')
    
    try:
        # The replicate is the second part of the filename.
        replicate_num = int(parts[1])
    except (IndexError, ValueError):
        replicate_num = 0

    try:
        # The object ID is the last part of the filename
        object_id = int(parts[-1])
    except (IndexError, ValueError):
        object_id = 0
        
    return (replicate_num, object_id)

def validate_arguments(args: argparse.Namespace) -> ValidatedArguments:
    # Validate model directory
    _validate_model(args.model)
    
    # Validate the input directory and get a list of vignette paths
    print(f"[CLASSIFICATION]: Loading vignettes from {args.input}")
    vignette_paths = _validate_input(args.input, sort_key=_get_vignette_sort_key)
    print(f"[CLASSIFICATION]: Found {len(vignette_paths)} vignettes")
    
    # Parse the vignette metadata from the input path and file name.
    # If we've reached this point, the input argument has been validated
    metadata = parse_vignette_metadata(Path(args.input))

    # Validate and create output path
    date_str = metadata["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(args.output, metadata["project"], date_str, metadata["cycle"], metadata["location"])
    output_path = _validate_output_path(output_path)

    # The detection CSV file must exist and be non-empty
    csv_path = os.path.join(output_path, CSV_FILENAME)
    detection_df = _validate_detection_csv(csv_path)

    # Return the validated arguments    
    return ValidatedArguments(
        vignette_paths=vignette_paths,
        output_path=output_path,
        model_path=args.model,
        metadata=metadata,
        batch_size=args.batch_size,
        input_size=args.input_size,
        input_depth=args.input_depth,
        categories=DEFAULT_CATEGORIES,
        csv_filename=CSV_FILENAME,
        detection_df=detection_df
    ) 