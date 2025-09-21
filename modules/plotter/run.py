"""
This script serves as the main entry point for the plotting module.
"""

import os
from types import SimpleNamespace
import pandas as pd

from modules.plotter.calculate_concentrations import (
    calculate_concentration_data,
    calculate_sizeclass_concentration_data,
)
from modules.plotter.plot_profile import PlotConfig, plot_single_profile
from modules.plotter.plot_length_profile import plot_length_profile
from modules.plotter.plot_size_profiles import plot_size_profiles
from modules.plotter.plotter_data import PlotterData, process_arguments


def _calculate_concentration(plotter_data: PlotterData) -> pd.DataFrame:
    """Calculates and saves the main concentration data."""
    print("[PLOTTER]: Calculating concentrations...")
    concentration_df = calculate_concentration_data(plotter_data)

    output_path = os.path.join(plotter_data.output_root, plotter_data.concentration_output_filename)
    concentration_df.to_csv(output_path, index=False, sep=plotter_data.csv_separator)

    print(f"[PLOTTER]: Concentration data saved to: {output_path}")
    return concentration_df

def _calculate_sizeclass_concentration(plotter_data: PlotterData) -> pd.DataFrame:
    """Calculates and saves the size-class concentration data."""
    print("[PLOTTER]: Calculating size-class concentrations...")
    sizeclass_concentration_df = calculate_sizeclass_concentration_data(
        plotter_data.object_data_df,
        plotter_data.bin_size,
        plotter_data.max_depth,
        plotter_data.img_depth,
        plotter_data.img_width,
        plotter_data.pixel_size_um
    )
    
    if sizeclass_concentration_df.empty:
        print("[PLOTTER]: No size-class concentration data was produced; skipping save.")
    else:
        output_path = os.path.join(plotter_data.output_root, plotter_data.sizeclass_concentration_filename)
        sizeclass_concentration_df.to_csv(output_path, index=False, sep=plotter_data.csv_separator)

        print(f"[PLOTTER]: Size-class concentration data saved to: {output_path}")
    return sizeclass_concentration_df


def run_plotter(run_config: SimpleNamespace, object_data_df: pd.DataFrame):
    """
    Executes the full plotting pipeline from concentration calculation to plot generation.
    """
    plotter_data = process_arguments(run_config, object_data_df)
        
    # Concentration and size-class concentration calculations
    concentration_df = _calculate_concentration(plotter_data)
    sizeclass_concentration_df = _calculate_sizeclass_concentration(plotter_data)

    # Plotting
    plot_config = PlotConfig(
        figsize=plotter_data.figsize,
        day_color=plotter_data.day_color,
        night_color=plotter_data.night_color,
        edge_color=plotter_data.edge_color,
        align=plotter_data.align,
        file_format=plotter_data.file_format
    )
    output_dir = plotter_data.output_root
    final_output_path = os.path.join(output_dir, 'plots')
    
    # Generate plots
    print("[PLOTTER]: Generating concentration plots...")
    plot_single_profile(concentration_df, output_dir, plot_config)
    
    print("[PLOTTER]: Generating length plots...")
    plot_length_profile(plotter_data, plot_config)

    print(f"[PLOTTER]: Plots for {output_dir} saved in {final_output_path}.")
    if not sizeclass_concentration_df.empty:
        print("[PLOTTER]: Generating size-class concentration profiles...")
        plot_size_profiles(sizeclass_concentration_df, output_dir, plot_config)
        
        print(f"[PLOTTER]: Size-class plots for {output_dir} saved in {final_output_path}.")
    else:
        print("[PLOTTER]: Size-class concentration DataFrame is empty; skipping size-class plots.")
