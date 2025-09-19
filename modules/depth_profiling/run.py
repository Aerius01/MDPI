from types import SimpleNamespace

import pandas as pd

from .depth_profile_data import process_arguments
from .profiler import profile_depths


def run_depth_profiling(run_config: SimpleNamespace) -> pd.DataFrame:
    """Run depth profiling and return the depth dataframe."""
    profiling_data = process_arguments(run_config)
    depth_df = profile_depths(profiling_data)
    return depth_df
