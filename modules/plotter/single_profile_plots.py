import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from modules.common.plot_utils import setup_plot_aesthetics, configure_axes, save_plot

def plot_single_profile(csv_path, output_path):
    # read day data
    data = pd.read_csv(csv_path)

    # get metadata from the first row
    first_row = data.iloc[0]
    date = first_row['date']
    time = first_row['time']
    repl = first_row['replicate']
    encl = os.path.basename(csv_path)[:3]

    # Group data by label and create a plot for each group
    for group, conc_data in data.groupby('label'):
        if conc_data.empty:
            continue

        # set plot title
        plot_title = f'{encl}_{group}_{date}_{time}_{repl}'

        # create plot
        fig, ax = plt.subplots(figsize=(10, 17.5))

        # set bar color based on time
        bar_color = 'white' if time == 'day' else 'grey'

        # create horizontal bar plot
        ax.barh(conc_data['depth'], conc_data['concentration'], height=conc_data['bin_size'], 
                edgecolor='black', color=bar_color, align='edge')

        # set axis labels and title
        setup_plot_aesthetics(ax, plot_title)

        # set axis ticks
        max_concentration = conc_data['concentration'].max() if not conc_data.empty else 0
        max_depth = conc_data['depth'].max() if not conc_data.empty else 0
        configure_axes(ax, max_depth, max_concentration, is_symmetric=False, depth_tick_step=1, conc_tick_step=100)

        # save plot
        save_plot(fig, output_path, plot_title)

def wrapper():
    parser = argparse.ArgumentParser(description='Generate single profile plots from concentration data.')
    parser.add_argument('-d', '--csv_path', type=str, help='Path to the input CSV file.', required=True)
    args = parser.parse_args()

    output_path = os.path.join(os.getcwd(), 'output', 'plots')
    
    plot_single_profile(args.csv_path, output_path)
    print(f'Plots for {args.csv_path} saved in the plot_data directory.') 
if __name__ == '__main__':
    wrapper()

