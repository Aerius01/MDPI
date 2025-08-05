import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from glob import glob
from modules.common.plot_utils import setup_plot_aesthetics, configure_axes, save_plot

def plot_day_night_profile(day_csv_path, night_csv_path, output_path):
    # read day and night data
    day_data = pd.read_csv(day_csv_path)
    night_data = pd.read_csv(night_csv_path)

    # get metadata from the first row of the day data
    first_row = day_data.iloc[0]
    date = first_row['date']
    repl = first_row['replicate']
    encl = os.path.basename(day_csv_path)[:3]

    # Combine data for group processing
    day_data['time_of_day'] = 'day'
    night_data['time_of_day'] = 'night'
    
    # Group data by label and create a plot for each group
    for group in day_data['label'].unique():
        conc_data_day = day_data[day_data['label'] == group]
        conc_data_night = night_data[night_data['label'] == group]

        if conc_data_day.empty and conc_data_night.empty:
            continue

        # set plot title
        plot_title = f'{encl}_{group}_{date}_{repl}'

        # create plot
        fig, ax = plt.subplots(figsize=(10, 17.5))

        # create horizontal bar plots
        ax.barh(conc_data_day['depth'], conc_data_day['concentration'], height=conc_data_day['bin_size'], 
                edgecolor='black', color='white', align='edge', label='Day')
        ax.barh(conc_data_night['depth'], -conc_data_night['concentration'], height=conc_data_night['bin_size'], 
                edgecolor='black', color='grey', align='edge', label='Night')

        # set axis labels and title
        setup_plot_aesthetics(ax, plot_title)
        ax.legend(fontsize=20)

        # set axis ticks
        max_concentration = max(conc_data_day['concentration'].max(), conc_data_night['concentration'].max()) if not conc_data_day.empty or not conc_data_night.empty else 0
        max_depth = max(conc_data_day['depth'].max(), conc_data_night['depth'].max()) if not conc_data_day.empty or not conc_data_night.empty else 0
        configure_axes(ax, max_depth, max_concentration, is_symmetric=True, depth_tick_step=1, conc_tick_step=100)

        # save plot
        save_plot(fig, output_path, plot_title)

def wrapper():
    parser = argparse.ArgumentParser(description='Generate day-night profile plots from concentration data.')
    parser.add_argument('-d', '--day_csv_path', type=str, help='Path to the day input CSV file.', required=True)
    parser.add_argument('-n', '--night_csv_path', type=str, help='Path to the night input CSV file.', required=True)
    args = parser.parse_args()

    output_path = os.path.join(os.getcwd(), 'output', 'plots')
    
    plot_day_night_profile(args.day_csv_path, args.night_csv_path, output_path)
    print(f'Plots for {args.day_csv_path} and {args.night_csv_path} saved in the plot_data directory.')

if __name__ == '__main__':
    wrapper()
