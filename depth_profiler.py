import os
import pandas as pd
import shutil
from datetime import datetime
from imutils import paths
from itertools import groupby
import re

# Pre-compile regex for efficiency
FILENAME_PATTERN = re.compile(r'(\d{8}_\d{6}\d{3})_(\d+)\.')

def _get_sort_key(path):
    filename = os.path.basename(path)
    match = FILENAME_PATTERN.search(filename)
    return int(match.group(2)) if match else 0

def profile_depths(input_path, output_path, timestep):
    print(f"[PROFILING] Starting depth matching...")
    
    # listing all images
    imagePaths = list(paths.list_images(input_path))
    
    # First group by directory, then sort each group
    imageGroups = [list(group) for key, group in groupby(sorted(imagePaths, key=os.path.dirname), os.path.dirname)]
    imageGroups = [sorted(group, key=_get_sort_key) for group in imageGroups]
    
    print(f"[PROFILING] Processing {len(imageGroups)} image groups...")

    for i, imageGroup in enumerate(imageGroups):
        # Get directory path and extract metadata
        imagePath = imageGroup[0]
        _, _, project, date, time, location, filename = imagePath.split(os.path.sep)
        root_dir = os.path.dirname(imagePath)

        print(f"[PROFILING] Processing group {i+1}/{len(imageGroups)}: {project}/{date}/{time}/{location} ({len(imageGroup)} images)")

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
        match = FILENAME_PATTERN.search(filename)
        if match:
            image_time = datetime.strptime(match.group(1), '%Y%m%d_%H%M%S%f')
            image_time = pd.to_datetime(image_time)
        else:
            raise ValueError(f"Could not parse datetime from filename: {filename}")

        # Pre-calculate all timestamps and depth values at once
        print(f"[PROFILING] Calculating timestamps and depth values...")
        timestamps = [image_time + (i * timestep) for i in range(len(imageGroup))]
        nearest_indices = csv.index.get_indexer(timestamps, method='nearest')
        depth_values = csv.iloc[nearest_indices]['depth'].values
        
        # Pre-generate all new filenames
        new_filenames = [f'{depth:.3f}_{project.replace("_","-")}_{date.replace("_","-")}_{time.replace("_","-")}_{location.replace("_","-")}.tiff' for depth in depth_values]
        new_filenames = [os.path.join(outputPath, filename) for filename in new_filenames]
        
        # Copy files with pre-generated names (batch operation)
        print(f"[PROFILING] Saving files...")
        [shutil.copy2(source_path, dest_path) for source_path, dest_path in zip(imageGroup, new_filenames)]

    print(f"[PROFILING] Depth matching completed successfully!")