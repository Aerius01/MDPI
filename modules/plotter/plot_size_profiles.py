import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from modules.plotter.plot_profile import PlotConfig
from modules.plotter.plot_utils import (
    setup_plot_aesthetics,
    configure_axes,
    save_plot,
    _calculate_nice_step,
)


def _validate_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    return [col for col in required_columns if col not in df.columns]


def plot_size_profiles(data: pd.DataFrame, output_path: str, config: PlotConfig, depth_tick_step: float = 1.0, conc_tick_step: float = 100.0):
    """
    Generate single profile plots per size class ('sizeclass' column).

    Expects columns:
      - recording_start_date, depth, bin_size, concentration, sizeclass
    The function creates a horizontal bar plot of concentration (ind/L) vs depth for each sizeclass.
    """
    if data.empty:
        print("[PLOTTER]: Size class input data is empty. Skipping size class plots.")
        return

    required_columns = [
        'recording_start_date', 'depth', 'bin_size', 'concentration', 'sizeclass'
    ]
    missing = _validate_columns(data, required_columns)
    if missing:
        print(f"[PLOTTER]: Input data missing required columns for size class plotting: {', '.join(missing)}. Skipping.")
        return

    # Metadata (assumes one sample per file)
    first_row = data.iloc[0]
    date = first_row['recording_start_date']

    # Group by species label and sizeclass and plot each
    label_key = 'label' if 'label' in data.columns else None
    group_keys = ['sizeclass'] if label_key is None else ['label', 'sizeclass']
    for keys, conc_data in data.groupby(group_keys):
        if conc_data.empty:
            continue

        if isinstance(keys, tuple):
            species_label, size_class = keys
        else:
            species_label, size_class = None, keys

        # Figure and aesthetics
        fig, ax = plt.subplots(figsize=config.figsize)
        bar_color = config.day_color

        # Horizontal bar plot
        ax.barh(
            conc_data['depth'],
            conc_data['concentration'],
            height=conc_data['bin_size'],
            edgecolor=config.edge_color,
            color=bar_color,
            align=config.align,
        )

        # Title and labels
        title_parts = [str(x) for x in [species_label, f"sizeclass_{size_class}", date] if x not in (None, '')]
        plot_title = '_'.join(title_parts)
        setup_plot_aesthetics(ax, plot_title)

        # Axes config
        raw_max_conc = conc_data['concentration'].max() if not conc_data.empty else 0
        step = _calculate_nice_step(raw_max_conc, 6)
        if step <= 0:
            step = conc_tick_step
        x_max = step * 6
        max_depth = conc_data['depth'].max() if not conc_data.empty else 0
        configure_axes(ax, max_depth, x_max, is_symmetric=False, depth_tick_step=depth_tick_step, conc_tick_step=step)

        # Save
        file_name = f"{plot_title}.{config.file_format}"
        label_folder = species_label if species_label is not None else 'sizeclass'
        sub_output_path = os.path.join(output_path, 'plots', str(label_folder))
        save_plot(fig, sub_output_path, file_name)

