import os
import pandas as pd
import shutil
from datetime import datetime
from constants import BASE_FILENAME_PATTERN, TIMESTEP

def profile_depths(image_group, output_path):
    print(f"[PROFILING]: Starting depth matching...")

    # Get directory path and extract metadata
    _, _, project, date, time, location, filename = image_group[0].split(os.path.sep)
    root_dir = os.path.dirname(image_group[0])

    print(f"[PROFILING]: Processing group: {project}/{date}/{time}/{location} ({len(image_group)} images)")

    # create output path
    outputPath = os.path.sep.join([output_path, project, date, time, location])
    os.makedirs(outputPath, exist_ok=True)

    # Get csv files from the image group directory (only once per group)
    csv_files = [f for f in os.listdir(root_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in directory: {root_dir}")
        
    # There should only be one .csv file present. Take the first one to be safe.
    csvPath = os.path.sep.join([root_dir, csv_files[0]])
    
    # Read and process CSV data (only once per group)
    csv = pd.read_csv(csvPath, sep=';', header=6, usecols=[0, 1], 
                    names=['time', 'depth'], index_col='time', 
                    skipfooter=1, engine='python')
    csv.index = pd.to_datetime(csv.index, format='%d.%m.%Y %H:%M:%S.%f')

    # Pre-process depth values, converting from German to international format
    csv['depth'] = csv['depth'].str.replace(',', '.').astype(float) * 10

    # Parse datetime from first image (only once per group)
    match = BASE_FILENAME_PATTERN.search(filename)
    if match:
        image_time = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S%f')
        image_time = pd.to_datetime(image_time)
    else:
        raise ValueError(f"Could not parse datetime from filename: {filename}")

    # Pre-calculate all timestamps and depth values at once
    print(f"[PROFILING]: Calculating timestamps and depth values...")
    timestamps = [image_time + (i * TIMESTEP) for i in range(len(image_group))]
    nearest_indices = csv.index.get_indexer(timestamps, method='nearest')
    depth_values = csv.iloc[nearest_indices]['depth'].values
    
    # Pre-generate all new filenames
    new_filenames = [f'{depth:.3f}_{project.replace("_","-")}_{date.replace("_","-")}_{time.replace("_","-")}_{location.replace("_","-")}.tiff' for depth in depth_values]
    new_filenames = [os.path.join(outputPath, filename) for filename in new_filenames]
    
    # Copy files with pre-generated names (batch operation)
    print(f"[PROFILING]: Saving files...")
    [shutil.copy2(source_path, dest_path) for source_path, dest_path in zip(image_group, new_filenames)]

    print(f"[PROFILING]: Depth matching completed successfully!")
    
    # Return new_filenames sorted by their name
    return sorted(new_filenames)