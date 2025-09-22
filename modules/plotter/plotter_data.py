from dataclasses import dataclass
from types import SimpleNamespace
import pandas as pd
from typing import Tuple
from modules.common.constants import CONSTANTS

# ─── P L O T T E R · C O N S T A N T S ──────────────────────────────────────────────────
# Plot aesthetics
FIGSIZE: Tuple[float, float] = (10, 17.5)
DAY_COLOR: str = 'white'
NIGHT_COLOR: str = 'grey'
EDGE_COLOR: str = 'black'
ALIGN: str = 'edge'
LEGEND_FONTSIZE: int = 20
FILE_FORMAT: str = 'png'
LENGTH_X_MAX_MM: float = 3.0 # crop x-axis of length plots to 3 mm for consistency

# Concentration calculation defaults
DEFAULT_BIN_SIZE = 0.10  # in meters
DEFAULT_MAX_DEPTH = 22.0  # in meters
CONCENTRATION_OUTPUT_FILENAME = "concentration_data.csv"
SIZECLASS_CONCENTRATION_FILENAME = "sizeclass_concentration_data.csv"

@dataclass
class PlotterData:
    object_data_df: pd.DataFrame
    output_root: str
    img_depth: float
    img_width: float
    max_depth: float
    bin_size: float
    concentration_output_filename: str
    sizeclass_concentration_filename: str
    figsize: Tuple[float, float]
    day_color: str
    night_color: str
    edge_color: str
    align: str
    file_format: str
    csv_separator: str
    pixel_size_um: float
    length_x_max_mm: float

def process_arguments(run_config: SimpleNamespace, object_data_df: pd.DataFrame) -> PlotterData:
    """
    Processes and validates the arguments for the plotter module.
    """
    if object_data_df.empty:
        raise ValueError("[PLOTTER]: Objects dataframe assembled by the classification module is empty! Cannot proceed with concentration calculation and plotting.")

    # Calculate pixel size in micrometers from image height (in cm) and pixels
    # (cm -> mm -> um)
    pixel_size_um = (run_config.image_height_cm * 10 / run_config.metadata["image_height_pixels"]) * 1000

    return PlotterData(
        object_data_df=object_data_df,
        output_root=run_config.output_root,
        img_depth=run_config.img_depth,
        img_width=run_config.img_width,
        max_depth=DEFAULT_MAX_DEPTH,
        bin_size=DEFAULT_BIN_SIZE,
        concentration_output_filename=CONCENTRATION_OUTPUT_FILENAME,
        sizeclass_concentration_filename=SIZECLASS_CONCENTRATION_FILENAME,
        figsize=FIGSIZE,
        day_color=DAY_COLOR,
        night_color=NIGHT_COLOR,
        edge_color=EDGE_COLOR,
        align=ALIGN,
        file_format=FILE_FORMAT,
        csv_separator=CONSTANTS.CSV_SEPARATOR,
        pixel_size_um=pixel_size_um,
        length_x_max_mm=LENGTH_X_MAX_MM
    )
