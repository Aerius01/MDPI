import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List

from modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot
from modules.plotter.constants import PLOTTING_CONSTANTS
from modules.plotter.plot_profile import PlotConfig


# Use centralized pixel size constant
PIXEL_SIZE_UM = PLOTTING_CONSTANTS.PIXEL_SIZE_UM  # micrometers per pixel
LENGTH_X_MAX_MM = 3.0  # crop x-axis at 3 mm for consistency


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


def plot_length_profile(data: pd.DataFrame, output_path: str, config: PlotConfig):
    """
    Generate and save length (mm) vs depth (m) scatter plots per taxonomic label.

    Expected columns in data:
      - recording_start_date, label, depth, MajorAxisLength
    """
    if data.empty:
        print("Input data is empty. No length plots will be generated.")
        return

    # Validate columns
    required_columns = [
        'recording_start_date', 'label', 'depth', 'MajorAxisLength'
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
    date = first_row['recording_start_date']

    # Groups to plot
    for group, group_df in data.groupby('label'):
        if group_df.empty:
            continue

        # Figure setup
        fig, ax = plt.subplots(figsize=config.figsize)
        point_color = config.day_color

        # Scatter plot (length on x, depth on y)
        ax.scatter(group_df['length_mm'], group_df['depth'], s=10, c=point_color, edgecolors=config.edge_color, linewidths=0.5)

        # Labels and title
        title_parts = [str(x) for x in [group, date] if x]
        plot_title = '_'.join(title_parts)
        setup_plot_aesthetics(ax, plot_title, xlabel='Length (mm)', ylabel='Depth (m)')

        # Axes configuration
        # Use fixed max x for consistency across plots
        max_length = LENGTH_X_MAX_MM
        max_depth = group_df['depth'].max() if not group_df.empty else 0
        configure_axes(ax, max_depth, max_length, is_symmetric=False, depth_tick_step=1, conc_tick_step=0.5)

        # Save
        base_prefix_parts = [str(x) for x in [group, date] if x]
        file_stem = f"{'_'.join(base_prefix_parts)}_length"
        file_name = f"{file_stem}.{config.file_format}"
        sub_output_path = os.path.join(output_path, 'plots', str(group))
        save_plot(fig, sub_output_path, file_name)