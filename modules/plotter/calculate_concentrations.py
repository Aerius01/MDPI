#!/usr/bin/env python3
"""
Python equivalent of MDPI_R/CalculateConcentrationData.R
Replicates the exact functionality for calculating concentration data from depth-based measurements.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Sequence

@dataclass
class ConcentrationConfig:
    """Configuration for concentration calculation."""
    max_depth: float
    bin_size: float
    output_file_name: str
    groups: List[str]
    img_depth: float
    img_width: float

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
    Calculates concentration [ind/l] from counts.
    """
    concentrations = np.round(counts / (bin_size * img_depth * img_width), decimals=1)
    return concentrations

def get_concentration_data(
    data: pd.DataFrame, 
    bin_size: float, 
    depth_bins: Sequence[float], 
    groups: List[str], 
    img_depth: float, 
    img_width: float
) -> pd.DataFrame:
    """
    Python equivalent of GetConcentrationData.R
    Calculates concentrations for each group and depth bin.
    """
    results_list = []
    
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
        
        # Calculate concentrations [ind/l]
        concentrations = calculate_concentration(counts, bin_size, img_depth, img_width)
        
        # Extract common metadata once
        metadata = {
            'project': data_subset['project'].iloc[0],
            'recording_start_date': data_subset['recording_start_date'].iloc[0],
            'cycle': data_subset['cycle'].iloc[0],
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
            'project', 'recording_start_date', 'cycle', 'depth', 
            'plot_depth', 'label', 'bin_size', 'concentration'
        ])

    concentration_data = pd.concat(results_list, ignore_index=True)
    
    # Remove rows with NA (equivalent to drop_na in R)
    concentration_data = concentration_data.dropna()
    
    return concentration_data

def calculate_concentration_data(data: pd.DataFrame, config: ConcentrationConfig) -> pd.DataFrame:
    """Main function for orchestrating concentration data calculation."""
    
    # Create sequence of bins - equivalent to seq(bin_size, max_depth, bin_size)
    depth_bins = np.arange(config.bin_size, config.max_depth + config.bin_size, config.bin_size)
        
    # Assign measurements to depth bins
    # Equivalent to sapply(data$depth, DepthBin, depth_bins)
    data['depth_bin'] = data['depth'].apply(lambda x: depth_bins[depth_bin(x, depth_bins)])
    
    # Calculate concentrations per group and bin
    concentration_data = get_concentration_data(
        data, config.bin_size, depth_bins, config.groups, config.img_depth, config.img_width
    )
    
    return concentration_data

if __name__ == "__main__":
    # --- Configuration ---
    BIN_SIZE = 0.1
    MAX_DEPTH = 18.0
    FILE_NAME = "concentration_data.csv"
    GROUPS = ['copepod', 'cladocera']
    IMG_DEPTH = 10.0
    IMG_WIDTH = 0.42
    REQUIRED_COLUMNS = [
        'project',
        'recording_start_date',
        'cycle',
        'replicate',
        'depth',
        'label'
    ]

    config = ConcentrationConfig(
        max_depth=MAX_DEPTH,
        bin_size=BIN_SIZE,
        output_file_name=FILE_NAME,
        groups=GROUPS,
        img_depth=IMG_DEPTH,
        img_width=IMG_WIDTH
    )

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Calculate concentration data from a single CSV file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input CSV file.")
    args = parser.parse_args()
    
    # --- Data Loading and Validation ---
    try:
        data = pd.read_csv(args.input, dtype={
            'project': str,
            'cycle': str,
            'replicate': str,
            'prediction': str,
            'label': str,
            'FileName': str
        })
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        print(f"Error: Input CSV file '{args.input}' is missing required columns: {', '.join(missing_columns)}")
        exit(1)

    # --- Calculation ---
    concentration_data = calculate_concentration_data(data, config)
    
    # --- Data Saving ---
    output_path = os.path.join(os.path.dirname(args.input), config.output_file_name)
    concentration_data.to_csv(output_path, index=False)
    print(f"[PLOTTER]: Concentration data saved to: {output_path}") 