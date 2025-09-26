import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot
from modules.plotter.plotter_data import PlotterData
from modules.plotter.plot_profile import PlotConfig


def plot_length_profile(plotter_data: PlotterData, config: PlotConfig):
    """
    Generate and save length (mm) vs depth (m) scatter plots per taxonomic label.

    Expected columns in data:
      - recording_start_date, label, depth, MajorAxisLength
    """
    # Ensure we operate on an independent DataFrame to avoid chained assignment issues
    data = plotter_data.object_data_df.copy(deep=True)
    output_path = plotter_data.output_root

    # Convert length from pixels to millimeters
    data['length_mm'] = (data['MajorAxisLength'] * plotter_data.pixel_size_um) / 1000.0

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
        max_length = plotter_data.length_x_max_mm
        max_depth = group_df['depth'].max() if not group_df.empty else 0
        configure_axes(ax, max_depth, max_length, is_symmetric=False, depth_tick_step=1, conc_tick_step=0.5)

        # Save
        base_prefix_parts = [str(x) for x in [group, date] if x]
        file_stem = f"{'_'.join(base_prefix_parts)}_length"
        file_name = f"{file_stem}.{config.file_format}"
        sub_output_path = os.path.join(output_path, 'plots', str(group))
        save_plot(fig, sub_output_path, file_name)