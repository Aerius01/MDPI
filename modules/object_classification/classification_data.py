from dataclasses import dataclass, field
import os
from pathlib import Path
import pandas as pd
from typing import Callable, Optional, List
from imutils import paths
from datetime import date, time
from types import SimpleNamespace

# ─── O B J E C T · C L A S S I F I C A T I O N · P A R A M E T E R S ──────────────
CLASSIFICATION_BATCH_SIZE: int = 32
CLASSIFICATION_INPUT_SIZE: int = 50
CLASSIFICATION_INPUT_DEPTH: int = 1
CLASSIFICATION_CATEGORIES: List[str] = ['cladocera', 'copepod', 'junk', 'rotifer'] # The available classification categories

# Hardcoded constants
DETECTED_OBJECTS_CSV_FILENAME = 'object_data.csv' # The filename of the detected objects CSV file, the output of the object detection module
OUTPUT_PKL_FILENAME = 'object_data.pkl' # The filename of the output classification pickle file (used for LabelChecker)
LEFT_JOIN_KEY = 'FileName' # The column name used to join the detection and classification dataframes
MODEL_CHECKPOINT_FILENAME = 'model.ckpt' # The filename of the model checkpoint file
MODEL_CHECKPOINT_EXTENSIONS = ['meta', 'index', 'data-00000-of-00001'] # The extensions of the required model checkpoint files

# The various metadata and CLI keys required by the classification module
EXPECTED_METADATA_KEYS = [
    "recording_start_date",
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
    recording_start_date: date
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

def validate_arguments(
    run_config: SimpleNamespace,
    vignettes_dir: str,
    batch_size: int,
    input_size: int,
    input_depth: int
) -> ClassificationData:
    vignette_paths = sorted(list(paths.list_images(vignettes_dir)), key=_get_vignette_sort_key)
    print(f"[CLASSIFICATION]: Found {len(vignette_paths)} vignettes")

    csv_path = os.path.join(run_config.output_root, DETECTED_OBJECTS_CSV_FILENAME)
    detection_df = pd.read_csv(csv_path, sep=None, engine='python')
    if detection_df.empty:
        raise ValueError(f"Detection CSV file is empty: {csv_path}")

    # Return the validated arguments    
    return ClassificationData(
        vignette_paths=vignette_paths,
        output_path=Path(run_config.output_root),
        model_path=run_config.model_dir,
        batch_size=batch_size,
        input_size=input_size,
        input_depth=input_depth,
        categories=CLASSIFICATION_CATEGORIES,
        csv_filename=DETECTED_OBJECTS_CSV_FILENAME,
        detection_df=detection_df,
        pkl_filename=OUTPUT_PKL_FILENAME,
        left_join_key=LEFT_JOIN_KEY,
        recording_start_date=run_config.metadata["recording_start_date"],
        total_replicates=run_config.metadata["total_replicates"],
        recording_start_time=run_config.metadata["recording_start_time"],
    ) 