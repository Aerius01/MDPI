from types import SimpleNamespace
import pandas as pd
from .detection_data import _validate_arguments
from .detector import _detect_objects

def run_detection(run_config: SimpleNamespace, depth_df: pd.DataFrame) -> pd.DataFrame:
    """Run detection."""
    detection_data = _validate_arguments(run_config, depth_df)
    object_data_df = _detect_objects(detection_data)
    return object_data_df