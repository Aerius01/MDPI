import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from dataclasses import dataclass
from modules.plotter.plot_utils import setup_plot_aesthetics, configure_axes, save_plot
from modules.plotter.constants import PLOTTING_CONSTANTS

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
    # get metadata from the first row (assuming it's consistent for the file)
    if data.empty:
        print("Input data is empty. No plots will be generated.")
        return

    first_row = data.iloc[0]
    project = first_row['project']
    date = first_row['recording_start_date']
    cycle = first_row['cycle']

    # Group data by label and create a plot for each group
    for group, conc_data in data.groupby('label'):
        if conc_data.empty:
            continue

        # set plot title and filename
        plot_title = f'{project}_{group}_{date}_{cycle}'
        file_name = f"{plot_title}.{config.file_format}"

        # create plot
        fig, ax = plt.subplots(figsize=config.figsize)

        # set bar color based on cycle
        bar_color = config.day_color if cycle == 'day' else config.night_color

        # create horizontal bar plot
        ax.barh(conc_data['depth'], conc_data['concentration'], height=conc_data['bin_size'], 
                edgecolor=config.edge_color, color=bar_color, align=config.align)

        # set axis labels and title
        setup_plot_aesthetics(ax, plot_title)

        # set axis ticks
        max_concentration = conc_data['concentration'].max() if not conc_data.empty else 0
        max_depth = conc_data['depth'].max() if not conc_data.empty else 0
        configure_axes(ax, max_depth, max_concentration, is_symmetric=False, depth_tick_step=1, conc_tick_step=100)

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
    FILE_FORMAT = PLOTTING_CONSTANTS.FILE_FORMAT
    REQUIRED_COLUMNS = PLOTTING_CONSTANTS.SINGLE_PROFILE_REQUIRED_COLUMNS

    config = PlotConfig(
        figsize=FIGSIZE,
        day_color=DAY_COLOR,
        night_color=NIGHT_COLOR,
        edge_color=EDGE_COLOR,
        align=ALIGN,
        file_format=FILE_FORMAT
    )

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Generate single profile plots from concentration data.')
    parser.add_argument('-i', '--csv_path', type=str, help='Path to the input CSV file.', required=True)
    args = parser.parse_args()

    # --- Data Loading ---
    try:
        input_csv = pd.read_csv(args.csv_path, sep=';', engine='python')
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.csv_path}")
        exit(1)

    # --- Validation ---
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in input_csv.columns]
    if missing_columns:
        print(f"Error: Input CSV file '{args.csv_path}' is missing required columns: {', '.join(missing_columns)}")
        exit(1)

    # --- Plotting ---
    output_path = os.path.dirname(args.csv_path)
    plot_single_profile(input_csv, output_path, config)
    
    print(f"[PLOTTER]: Plots for {args.csv_path} saved in {output_path}.") 

