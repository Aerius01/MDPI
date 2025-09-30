import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from pipeline.modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot, _calculate_nice_step

@dataclass
class PlotConfig:
    """Configuration for plotting profiles."""
    figsize: tuple
    day_color: str
    night_color: str
    edge_color: str
    align: str
    file_format: str

def plot_single_profile(data: pd.DataFrame, output_path: str, config: PlotConfig):
    """
    Generates and saves profile plots for each group in the dataframe.
    """
    first_row = data.iloc[0]
    date = first_row['recording_start_date']

    # Group data by label and create a plot for each group
    for group, conc_data in data.groupby('label'):
        if conc_data.empty:
            continue

        # set plot title and filename
        plot_title = f'{group}_{date}'
        file_name = f"{plot_title}_concentration.{config.file_format}"

        # create plot
        fig, ax = plt.subplots(figsize=config.figsize)

        # set bar color based on cycle
        bar_color = config.day_color

        # create horizontal bar plot
        ax.barh(conc_data['depth'], conc_data['concentration'], height=conc_data['bin_size'], 
                edgecolor=config.edge_color, color=bar_color, align=config.align)

        # set axis labels and title
        setup_plot_aesthetics(ax, plot_title)

        # set axis ticks (standardize to 5 ticks: 0, 25%, 50%, 75%, 100%)
        raw_max_concentration = conc_data['concentration'].max() if not conc_data.empty else 0
        step = _calculate_nice_step(raw_max_concentration, 4)
        if step <= 0:
            step = 1
        x_max = step * 4
        max_depth = conc_data['depth'].max() if not conc_data.empty else 0
        configure_axes(ax, max_depth, x_max, is_symmetric=False, depth_tick_step=1, conc_tick_step=step)

        # save plot into standardized subfolder by label
        sub_output_path = os.path.join(output_path, 'plots', str(group))
        save_plot(fig, sub_output_path, file_name)
