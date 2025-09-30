from dataclasses import dataclass
import os
from pathlib import Path
import pandas as pd
from typing import List
from imutils import paths
from datetime import date, time
from types import SimpleNamespace
from pipeline.modules.common.constants import CONSTANTS

# ─── O B J E C T · C L A S S I F I C A T I O N · P A R A M E T E R S ──────────────
CLASSIFICATION_BATCH_SIZE: int = 32
CLASSIFICATION_INPUT_SIZE: int = 50
CLASSIFICATION_INPUT_DEPTH: int = 1
CLASSIFICATION_CATEGORIES: List[str] = ['cladocera', 'copepod', 'junk', 'rotifer'] # The available classification categories

# Hardcoded constants
OUTPUT_CSV_FILENAME = 'object_data.csv' # The filename of the output objects CSV file
OUTPUT_PKL_FILENAME = 'object_data.pkl' # The filename of the output classification pickle file (used for LabelChecker)
LEFT_JOIN_KEY = 'FileName' # The column name used to join the detection and classification dataframes

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
    csv_separator: str
    detection_df: pd.DataFrame
    pkl_filename: str
    left_join_key: str
    recording_start_date: date
    total_replicates: int
    recording_start_time: time

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

def _validate_arguments(
    run_config: SimpleNamespace,
    object_data_df: pd.DataFrame,
) -> ClassificationData:
    vignettes_dir = os.path.join(run_config.output_root, CONSTANTS.VIGNETTES_DIR_NAME)
    vignette_paths = sorted(list(paths.list_images(vignettes_dir)), key=_get_vignette_sort_key)
    print(f"[CLASSIFICATION]: Found {len(vignette_paths)} vignettes")

    if object_data_df.empty:
        raise ValueError(f"Input object_data_df is empty")

    # Return the validated arguments    
    return ClassificationData(
        vignette_paths=vignette_paths,
        output_path=Path(run_config.output_root),
        model_path=run_config.model_dir,
        batch_size=CLASSIFICATION_BATCH_SIZE,
        input_size=CLASSIFICATION_INPUT_SIZE,
        input_depth=CLASSIFICATION_INPUT_DEPTH,
        categories=CLASSIFICATION_CATEGORIES,
        csv_filename=OUTPUT_CSV_FILENAME,
        csv_separator=CONSTANTS.CSV_SEPARATOR,
        detection_df=object_data_df,
        pkl_filename=OUTPUT_PKL_FILENAME,
        left_join_key=LEFT_JOIN_KEY,
        recording_start_date=run_config.metadata["recording_start_date"],
        total_replicates=run_config.metadata["total_replicates"],
        recording_start_time=run_config.metadata["recording_start_time"],
    ) 