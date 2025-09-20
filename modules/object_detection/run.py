from types import SimpleNamespace
import pandas as pd

from .detection_data import (
    validate_arguments,
)
from .detector import detect_objects


def run_detection(run_config: SimpleNamespace, flatfield_dir: str, depth_df: pd.DataFrame) -> str:
    """Run detection and return the vignettes output directory."""
    detection_data = validate_arguments(run_config, flatfield_dir, depth_df)

    detect_objects(detection_data)
    return detection_data.output_path