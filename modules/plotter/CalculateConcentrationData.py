#!/usr/bin/env python3
"""
Python equivalent of MDPI_R/CalculateConcentrationData.R
Replicates the exact functionality for calculating concentration data from depth-based measurements.
"""

import os
import glob
import pandas as pd
import numpy as np

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

def calculate_concentration(counts, bin_size, img_depth=10, img_width=0.42):
    """
    Python equivalent of CalculateConcentration.R
    Calculates concentration [ind/l] from counts.
    """
    concentrations = np.round(counts / (bin_size * img_depth * img_width), decimals=1)
    return concentrations

def get_concentration_data(data, bin_size, depth_bins, groups):
    """
    Python equivalent of GetConcentrationData.R
    Calculates concentrations for each group and depth bin.
    """
    # Create empty DataFrame with same structure as R tibble
    concentration_data = pd.DataFrame(columns=[
        'project', 'date', 'time', 'replicate', 'depth', 
        'plot_depth', 'label', 'bin_size', 'concentration'
    ])
    
    # Calculate concentrations per group
    for group in groups:
        # Count individuals per depth bin
        data_subset = data[data['label'] == group]
        counts = np.zeros(len(depth_bins), dtype=int)
        
        for i, bin_depth in enumerate(depth_bins):
            count = len(data_subset[data_subset['depth_bin'] == bin_depth])
            counts[i] = count
        
        # Calculate concentrations [ind/l]
        concentrations = calculate_concentration(counts, bin_size)
        
        # Extract common metadata once
        metadata = {
            'project': data_subset['project'].iloc[0],
            'date': data_subset['date'].iloc[0],
            'time': data_subset['time'].iloc[0],
            'replicate': data_subset['replicate'].iloc[0]
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
        
        # Append to main DataFrame
        concentration_data = pd.concat([concentration_data, results_df], ignore_index=True)
    
    # Remove rows with NA (equivalent to drop_na in R)
    concentration_data = concentration_data.dropna()
    
    return concentration_data

def calculate_concentration_data(csv_paths, output_path, max_depth, bin_size):
    """Main function replicating the R script logic exactly."""
    
    # Create sequence of bins - equivalent to seq(bin_size, max_depth, bin_size)
    depth_bins = np.arange(bin_size, max_depth + bin_size, bin_size)
    
    # Info output - equivalent to cat(paste(...))
    print(f"Maximum plot depth: {max_depth}; with bin size: {bin_size};")
    print("To change bin_size and, or max_depth parameters, changes the values set at lines 31 & 32")
    
    # Process each CSV file
    for csv in csv_paths:
        # Gathering day-night data
        # Read day data - equivalent to read_csv with col_types
        data = pd.read_csv(csv, dtype={
            'project': str,
            'time': str,
            'replicate': str,
            'prediction': str,
            'label': str,
            'Filename': str
        })
        
        # Get date and replicate
        project = data['project'].iloc[0]
        date = data['date'].iloc[0]
        time = data['time'].iloc[0]
        repl = data['replicate'].iloc[0]
        
        # Christian - equivalent to substring(basename(csv),1,3)
        encl = os.path.basename(csv)[:3]
        
        # Assign measurements to depth bins
        # Equivalent to sapply(data$depth, DepthBin, depth_bins)
        data['depth_bin'] = data['depth'].apply(lambda x: depth_bins[depth_bin(x, depth_bins)])
        
        # Calculate concentrations per group and bin
        groups = ['copepod', 'cladocera']
        concentration_data = get_concentration_data(data, bin_size, depth_bins, groups)
        
        # Save concentration data
        # Set paths
        CONC_SAVE_PATH = os.path.join(output_path, project, 'concentration_data', 
                                     str(bin_size), str(date), time)
        os.makedirs(CONC_SAVE_PATH, exist_ok=True)
        
        # Filename - Christian's version
        filename = f"{encl}_{date}_{time}_{repl}_concentration_data.csv"
        
        # Save data - equivalent to write_csv
        concentration_data.to_csv(os.path.join(CONC_SAVE_PATH, filename), index=False)

def wrapper():
    directory = '/home/david-james/Desktop/04-MDPI/MDPI/output/restructured'
    csv_paths = glob.glob(os.path.join(directory, '**/*.csv'), recursive=True)
    output_path = os.path.join('.', 'output', 'concentrations')
    bin_size = 0.1
    max_depth = 18.0
    calculate_concentration_data(csv_paths, output_path, max_depth, bin_size)

if __name__ == "__main__":
    wrapper() 