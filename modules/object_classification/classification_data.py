from dataclasses import dataclass
import os
from pathlib import Path
import pandas as pd
from modules.common.constants import CONSTANTS
from typing import Callable, Optional
from imutils import paths
from datetime import date, time

# Destructure constants
DEFAULT_CATEGORIES = CONSTANTS.CLASSIFICATION_CATEGORIES

# Hardcoded constants
CSV_FILENAME = 'object_data.csv'
PKL_FILENAME = 'object_data.pkl'
LEFT_JOIN_KEY = 'FileName'
MODEL_CHECKPOINT_FILENAME = 'model.ckpt'
MODEL_CHECKPOINT_EXTENSIONS = ['meta', 'index', 'data-00000-of-00001']
EXPECTED_METADATA_KEYS = [
    "project",
    "recording_start_date",
    "cycle",
    "location",
    "total_replicates",
    "recording_start_time",
]
EXPECTED_CLI_KEYS = [
    "model", 
    "input", 
    "output", 
    "batch_size", 
    "input_size", 
    "input_depth"
]

@dataclass
class ClassificationData:
    vignette_paths: list
    output_path: Path
    model_path: str
    batch_size: int
    input_size: int
    input_depth: int
    categories: list
    csv_filename: str
    detection_df: pd.DataFrame
    pkl_filename: str
    left_join_key: str
    project: str
    recording_start_date: date
    cycle: str
    location: str
    total_replicates: int
    recording_start_time: time

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

def validate_arguments(**kwargs) -> ClassificationData:
    # Check for required arguments
    expected_keys = EXPECTED_METADATA_KEYS + EXPECTED_CLI_KEYS
    missing_keys = [key for key in expected_keys if key not in kwargs]
    if missing_keys:
        raise ValueError(f"Missing required arguments: {', '.join(missing_keys)}")
        
    # Validate model directory
    _validate_model(kwargs["model"])
    
    # Validate the input directory and get a list of vignette paths
    print(f"[CLASSIFICATION]: Loading vignettes from {kwargs['input']}")
    vignette_paths = _validate_input(kwargs["input"], sort_key=_get_vignette_sort_key)
    print(f"[CLASSIFICATION]: Found {len(vignette_paths)} vignettes")

    # Validate and create output path
    date_str = kwargs["recording_start_date"].strftime("%Y%m%d")
    output_path = os.path.join(kwargs["output"], kwargs["project"], date_str, kwargs["cycle"], kwargs["location"])
    output_path = Path(_validate_output_path(output_path))

    # The detection CSV file must exist and be non-empty
    csv_path = os.path.join(output_path, CSV_FILENAME)
    detection_df = _validate_detection_csv(csv_path)

    # Return the validated arguments    
    return ClassificationData(
        vignette_paths=vignette_paths,
        output_path=output_path,
        model_path=kwargs["model"],
        batch_size=kwargs["batch_size"],
        input_size=kwargs["input_size"],
        input_depth=kwargs["input_depth"],
        categories=DEFAULT_CATEGORIES,
        csv_filename=CSV_FILENAME,
        detection_df=detection_df,
        pkl_filename=PKL_FILENAME,
        left_join_key=LEFT_JOIN_KEY,
        project=kwargs["project"],
        recording_start_date=kwargs["recording_start_date"],
        cycle=kwargs["cycle"],
        location=kwargs["location"],
        total_replicates=kwargs["total_replicates"],
        recording_start_time=kwargs["recording_start_time"],
    ) 