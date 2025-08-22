import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from dataclasses import dataclass
from modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot
from modules.common.constants import PLOTTING_CONSTANTS

@dataclass
class DayNightPlotConfig:
    """Configuration for plotting day-night profiles."""
    figsize: tuple
    day_color: str
    night_color: str
    edge_color: str
    align: str
    legend_fontsize: int
    file_format: str

def plot_day_night_profile(day_data: pd.DataFrame, night_data: pd.DataFrame, output_path: str, config: DayNightPlotConfig):
    """
    Generates and saves day-night profile plots for each group.
    """
    # get metadata from the first row of the day data
    if day_data.empty or night_data.empty:
        print("One or both input dataframes are empty. No plots will be generated.")
        return

    first_row = day_data.iloc[0]
    project = first_row['project']
    date = first_row['recording_start_date']

    # Group data by label and create a plot for each group
    for group in day_data['label'].unique():
        conc_data_day = day_data[day_data['label'] == group]
        conc_data_night = night_data[night_data['label'] == group]

        if conc_data_day.empty and conc_data_night.empty:
            continue

        # set plot title and filename
        plot_title = f'{project}_{group}_{date}'
        file_name = f"{plot_title}.{config.file_format}"

        # create plot
        fig, ax = plt.subplots(figsize=config.figsize)

        # create horizontal bar plots
        ax.barh(conc_data_day['depth'], conc_data_day['concentration'], height=conc_data_day['bin_size'], 
                edgecolor=config.edge_color, color=config.day_color, align=config.align, label='Day')
        ax.barh(conc_data_night['depth'], -conc_data_night['concentration'], height=conc_data_night['bin_size'], 
                edgecolor=config.edge_color, color=config.night_color, align=config.align, label='Night')

        # set axis labels and title
        setup_plot_aesthetics(ax, plot_title)
        ax.legend(fontsize=config.legend_fontsize)

        # set axis ticks
        max_concentration = max(conc_data_day['concentration'].max(), conc_data_night['concentration'].max()) if not conc_data_day.empty or not conc_data_night.empty else 0
        max_depth = max(conc_data_day['depth'].max(), conc_data_night['depth'].max()) if not conc_data_day.empty or not conc_data_night.empty else 0
        configure_axes(ax, max_depth, max_concentration, is_symmetric=True, depth_tick_step=1, conc_tick_step=100)

        # save plot
        save_plot(fig, output_path, file_name)

if __name__ == '__main__':
    # --- Configuration ---
    # Destructure constants from the central constants file
    FIGSIZE = PLOTTING_CONSTANTS.FIGSIZE
    DAY_COLOR = PLOTTING_CONSTANTS.DAY_COLOR
    NIGHT_COLOR = PLOTTING_CONSTANTS.NIGHT_COLOR
    EDGE_COLOR = PLOTTING_CONSTANTS.EDGE_COLOR
    ALIGN = PLOTTING_CONSTANTS.ALIGN
    LEGEND_FONTSIZE = PLOTTING_CONSTANTS.LEGEND_FONTSIZE
    FILE_FORMAT = PLOTTING_CONSTANTS.FILE_FORMAT
    REQUIRED_COLUMNS = PLOTTING_CONSTANTS.DAY_NIGHT_PROFILE_REQUIRED_COLUMNS

    config = DayNightPlotConfig(
        figsize=FIGSIZE,
        day_color=DAY_COLOR,
        night_color=NIGHT_COLOR,
        edge_color=EDGE_COLOR,
        align=ALIGN,
        legend_fontsize=LEGEND_FONTSIZE,
        file_format=FILE_FORMAT
    )

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Generate day-night profile plots from concentration data.')
    parser.add_argument('-d', '--day_csv_path', type=str, help='Path to the day input CSV file.', required=True)
    parser.add_argument('-n', '--night_csv_path', type=str, help='Path to the night input CSV file.', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Path for the output plots.', default='./output')
    args = parser.parse_args()

    # --- Data Loading ---
    try:
        day_data = pd.read_csv(args.day_csv_path)
        night_data = pd.read_csv(args.night_csv_path)
    except FileNotFoundError as e:
        print(f"Error: Input file not found at {e.filename}")
        exit(1)

    # Validate that the day and night csv files have the required columns
    missing_day_cols = [col for col in REQUIRED_COLUMNS if col not in day_data.columns]
    if missing_day_cols:
        print(f"Error: Day CSV file '{args.day_csv_path}' is missing required columns: {', '.join(missing_day_cols)}")
        exit(1)

    missing_night_cols = [col for col in REQUIRED_COLUMNS if col not in night_data.columns]
    if missing_night_cols:
        print(f"Error: Night CSV file '{args.night_csv_path}' is missing required columns: {', '.join(missing_night_cols)}")
        exit(1)

    # Create output directory if it doesn't exist
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    plot_day_night_profile(day_data, night_data, output_path, config)
    print(f"[PLOTTER]: Plots for {args.day_csv_path} and {args.night_csv_path} saved in {output_path}.")
