#!/usr/bin/env python3
"""
Python equivalent of MDPI_R/CalculateConcentrationData.R
Replicates the exact functionality for calculating concentration data from depth-based measurements.
"""

import pandas as pd
import numpy as np
from typing import Sequence, List
from modules.plotter.plotter_data import PlotterData


# Custom functions equivalent to R tools

def depth_bin(depth, bin_sequence):
    """
    Python equivalent of DepthBin.R
    Assigns a depth value to the nearest bin in the sequence.
    Returns an index.
    """
    bin_sequence = np.array(bin_sequence)
    min_idx = np.argmin(np.abs(depth - bin_sequence))
    
    if bin_sequence[min_idx] > depth:
        return min_idx
    else:
        if min_idx + 1 < len(bin_sequence):
            return min_idx + 1
        else:
            return min_idx

def depth_bin_offset(bin_sequence):
    """
    Python equivalent of DepthBinOffset.R
    Offsets bin depth by half bin_size to correct for barplot behaviour.
    """
    bin_sequence = np.array(bin_sequence)
    plot_depth_bin = [bin_sequence[0] * 0.5]
    
    for i in range(1, len(bin_sequence)):
        offset = bin_sequence[i-1] + (bin_sequence[i] - bin_sequence[i-1]) * 0.5
        plot_depth_bin.append(offset)
    
    return plot_depth_bin

def calculate_concentration(counts: np.ndarray, bin_size: float, img_depth: float, img_width: float) -> np.ndarray:
    """
    Python equivalent of CalculateConcentration.R
    Calculates concentration [ind/L] from counts.
    The volume is calculated in liters (bin_size * img_depth * img_width == dm^3).
    """
    bin_size_dm = bin_size * 10 # convert to decimeters
    volume_L = bin_size_dm * img_depth * img_width
    concentrations = np.round(counts / volume_L, decimals=1)
    return concentrations

def get_concentration_data(
    data: pd.DataFrame, 
    bin_size: float, 
    depth_bins: Sequence[float], 
    img_depth: float, 
    img_width: float
) -> pd.DataFrame:
    """
    Python equivalent of GetConcentrationData.R
    Calculates concentrations for each group and depth bin.
    """
    results_list = []
    
    groups = [group for group in data['label'].unique()]
    
    # Calculate concentrations per group
    for group in groups:
        # Count individuals per depth bin
        data_subset = data[data['label'] == group]
        if data_subset.empty:
            continue
        counts = np.zeros(len(depth_bins), dtype=int)
        
        for i, bin_depth in enumerate(depth_bins):
            count = len(data_subset[data_subset['depth_bin'] == bin_depth])
            counts[i] = count
        
        # Calculate concentrations [ind/L]
        concentrations = calculate_concentration(counts, bin_size, img_depth, img_width)
        
        # Extract common metadata once
        metadata = {
            'recording_start_date': data_subset['recording_start_date'].iloc[0],
        }
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            **{col: [val] * len(depth_bins) for col, val in metadata.items()},
            'depth': np.round(depth_bins, decimals=2),
            'plot_depth': np.round(depth_bin_offset(depth_bins), decimals=2),
            'label': [group] * len(depth_bins),
            'bin_size': [bin_size] * len(depth_bins),
            'concentration': concentrations
        })
        
        results_list.append(results_df)
    
    if not results_list:
        return pd.DataFrame(columns=[
            'recording_start_date', 'depth', 
            'plot_depth', 'label', 'bin_size', 'concentration'
        ])

    concentration_data = pd.concat(results_list, ignore_index=True)
    
    # Remove rows with NA (equivalent to drop_na in R)
    concentration_data = concentration_data.dropna()
    
    return concentration_data

def calculate_concentration_data(plotter_data: PlotterData) -> pd.DataFrame:
    """Main function for orchestrating concentration data calculation."""
    # Work on a deep copy to avoid chained assignment warnings
    data = plotter_data.object_data_df.copy(deep=True)
    
    # Create sequence of bins - equivalent to seq(bin_size, max_depth, bin_size)
    depth_bins = np.arange(plotter_data.bin_size, plotter_data.max_depth + plotter_data.bin_size, plotter_data.bin_size)
        
    # Assign measurements to depth bins
    # Equivalent to sapply(data$depth, DepthBin, depth_bins)
    # Single-step assignment using .loc to avoid chained assignment semantics
    data.loc[:, 'depth_bin'] = data['depth'].apply(lambda x: depth_bins[depth_bin(x, depth_bins)])
    
    # Calculate concentrations per group and bin
    concentration_data = get_concentration_data(
        data, plotter_data.bin_size, depth_bins, plotter_data.img_depth, plotter_data.img_width
    )
    
    return concentration_data


# ------------------ Size class concentration calculation ------------------

def _assign_size_classes(group_df: pd.DataFrame, pixel_size_um: float, num_classes: int = 3) -> pd.Series:
    """
    Assign quantile-based size classes within a taxonomic label group using EquivDiameter.
    Returns a categorical series with labels '1'..'num_classes'.
    """
    # Prefer EquivDiameter; fallback to MajorAxisLength if needed
    if 'EquivDiameter' in group_df.columns:
        size_metric = (group_df['EquivDiameter'] * pixel_size_um) / 1000.0
    elif 'MajorAxisLength' in group_df.columns:
        size_metric = (group_df['MajorAxisLength'] * pixel_size_um) / 1000.0
    else:
        # No metric available; everything goes to class '1'
        return pd.Series(['1'] * len(group_df), index=group_df.index, dtype=str)

    # Quantile-based binning akin to Hmisc::cut2(..., g=3)
    try:
        classes = pd.qcut(size_metric, q=num_classes, labels=[str(i) for i in range(1, num_classes + 1)])
        return classes.astype(str)
    except Exception:
        # Fallback to equal-width if not enough unique values
        try:
            classes = pd.cut(size_metric, bins=num_classes, labels=[str(i) for i in range(1, num_classes + 1)], include_lowest=True)
            return classes.astype(str)
        except Exception:
            return pd.Series(['1'] * len(group_df), index=group_df.index, dtype=str)


def calculate_sizeclass_concentration_data(
    object_data_df: pd.DataFrame, 
    bin_size: float, 
    max_depth: float, 
    img_depth: float, 
    img_width: float, 
    pixel_size_um: float,
    groups: List[str] = None
) -> pd.DataFrame:
    """
    Calculate concentration data aggregated by size classes (1..3) within each taxonomic label.
    Mirrors MDPI-Ashton/tools/io/GetConcentrationDataSizeClass.R using Python.
    """
    # Work on a deep copy to avoid chained assignment warnings
    data = object_data_df.copy(deep=True)
    # Create sequence of bins - equivalent to seq(bin_size, max_depth, bin_size)
    depth_bins = np.arange(bin_size, max_depth + bin_size, bin_size)

    # Assign measurements to depth bins
    data.loc[:, 'depth_bin'] = data['depth'].apply(lambda x: depth_bins[depth_bin(x, depth_bins)])

    # Ensure sizeclass exists; if not, derive within each label group using quantiles
    if 'sizeclass' not in data.columns:
        data.loc[:, 'sizeclass'] = (
            data.groupby('label', group_keys=False)
                .apply(lambda g: _assign_size_classes(g, pixel_size_um), include_groups=False)
        )

    # Determine which size classes to use
    if groups is None:
        groups = ['1', '2', '3']

    concentration_rows = []

    # Calculate concentrations per label and size class
    labels = data['label'].unique().tolist() if 'label' in data.columns else [None]
    for label_value in labels:
        label_subset = data if label_value is None else data[data['label'] == label_value]
        if label_subset.empty:
            continue

        for size_group in groups:
            subset = label_subset[label_subset['sizeclass'] == size_group]
            if subset.empty:
                continue

            counts = np.zeros(len(depth_bins), dtype=int)
            for i, bin_depth in enumerate(depth_bins):
                counts[i] = (subset['depth_bin'] == bin_depth).sum()

            concentrations = calculate_concentration(counts, bin_size, img_depth, img_width)

            meta_date = subset['recording_start_date'].iloc[0] if 'recording_start_date' in subset.columns else subset.get('date', pd.Series([None])).iloc[0]

            row_dict = {
                'recording_start_date': [meta_date] * len(depth_bins),
                'depth': np.round(depth_bins, 2),
                'plot_depth': np.round(depth_bin_offset(depth_bins), 2),
                'sizeclass': [size_group] * len(depth_bins),
                'bin_size': [bin_size] * len(depth_bins),
                'concentration': concentrations,
            }
            if label_value is not None:
                row_dict['label'] = [label_value] * len(depth_bins)

            concentration_rows.append(pd.DataFrame(row_dict))

    if not concentration_rows:
        return pd.DataFrame(columns=['recording_start_date','label','depth','plot_depth','sizeclass','bin_size','concentration'])

    result = pd.concat(concentration_rows, ignore_index=True)
    result = result.dropna()
    return result