from types import SimpleNamespace
import pandas as pd
from .flatfielding_data import process_arguments
from .flatfielding import flatfield_images


def run_flatfielding(run_config: SimpleNamespace, depth_df: pd.DataFrame):
    """Run flatfielding."""
    data = process_arguments(run_config, depth_df)
    flatfield_images(data)
