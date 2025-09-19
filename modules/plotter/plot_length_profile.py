import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List

from modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot
from modules.plotter.constants import PLOTTING_CONSTANTS
from modules.plotter.plot_profile import PlotConfig


PIXEL_SIZE_UM = 20.9  # micrometers per pixel
LENGTH_X_MAX_MM = 3.0  # crop x-axis at 3 mm for consistency


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


def plot_length_profile(data: pd.DataFrame, output_path: str, config: PlotConfig):
    """
    Generate and save length (mm) vs depth (m) scatter plots per taxonomic label.

    Expected columns in data:
      - project, recording_start_date, cycle, label, depth, MajorAxisLength
      - location (optional, used for file naming if present)
    """
    if data.empty:
        print("Input data is empty. No length plots will be generated.")
        return

    # Validate columns
    required_columns = [
        'project', 'recording_start_date', 'cycle', 'label', 'depth', 'MajorAxisLength'
    ]
    missing = _validate_columns(data, required_columns)
    if missing:
        print(f"Error: Input data is missing required columns for length plotting: {', '.join(missing)}")
        return

    # Convert length from pixels to millimeters
    data = data.copy()
    data['length_mm'] = (data['MajorAxisLength'] * PIXEL_SIZE_UM) / 1000.0

    # Metadata (assume single sample per file)
    first_row = data.iloc[0]
    project = first_row['project']
    date = first_row['recording_start_date']
    cycle = first_row['cycle']
    location = first_row['location'] if 'location' in data.columns else None

    # Groups to plot
    for group, group_df in data.groupby('label'):
        if group_df.empty:
            continue

        # Figure setup
        fig, ax = plt.subplots(figsize=config.figsize)
        point_color = config.day_color if cycle == 'day' else config.night_color

        # Scatter plot (length on x, depth on y)
        ax.scatter(group_df['length_mm'], group_df['depth'], s=10, c=point_color, edgecolors=config.edge_color, linewidths=0.5)

        # Labels and title
        title_parts = [str(x) for x in [project, group, date, cycle, location] if x]
        plot_title = '_'.join(title_parts)
        setup_plot_aesthetics(ax, plot_title, xlabel='Length (mm)', ylabel='Depth (m)')

        # Axes configuration
        # Use fixed max x for consistency across plots
        max_length = LENGTH_X_MAX_MM
        max_depth = group_df['depth'].max() if not group_df.empty else 0
        configure_axes(ax, max_depth, max_length, is_symmetric=False, depth_tick_step=1, conc_tick_step=0.5)

        # Save
        base_prefix_parts = [str(x) for x in [project, group, date, cycle] if x]
        file_stem = f"{'_'.join(base_prefix_parts)}_length"
        if location is not None:
            file_stem = f"{file_stem}_{location}"
        file_name = f"{file_stem}.{config.file_format}"
        sub_output_path = os.path.join(output_path, 'plots')
        save_plot(fig, sub_output_path, file_name)

FIGSIZE = PLOTTING_CONSTANTS.FIGSIZE
DAY_COLOR = PLOTTING_CONSTANTS.DAY_COLOR
NIGHT_COLOR = PLOTTING_CONSTANTS.NIGHT_COLOR
EDGE_COLOR = PLOTTING_CONSTANTS.EDGE_COLOR
ALIGN = PLOTTING_CONSTANTS.ALIGN
FILE_FORMAT = PLOTTING_CONSTANTS.FILE_FORMAT

config = PlotConfig(
    figsize=FIGSIZE,
    day_color=DAY_COLOR,
    night_color=NIGHT_COLOR,
    edge_color=EDGE_COLOR,
    align=ALIGN,
    file_format=FILE_FORMAT
)
