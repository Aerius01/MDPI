import argparse
from pathlib import Path
import pandas as pd
from modules.common import match_files_by_lcs
from typing import List

def data_restructuring(data, column, separator=r'([/\\_])', column_names=None, treatment=False):
    """
    Python equivalent of the R DataRestructuring function.
    
    Args:
        data: DataFrame with MDPI data
        column: Column name to separate
        separator: Regex pattern for splitting
        column_names: List of column names for split parts (NA values will be dropped)
        treatment: Whether to add treatment column
    """
    if column_names is None:
        column_names = [None, None, 'project', 'date', 'time', None, None, 'replicate', 'depth', None]
    
    # Extract filename from the full path and split it
    # The filename structure is: depth_project_date_time_location_replicate.jpeg
    data['filename'] = data[column].apply(lambda x: Path(x).name)
    
    # Remove any file extension and split the filename using underscore
    filename_parts = data['filename'].str.replace(r'\.[^.]*$', '', regex=True).str.split('_', expand=True)
    
    # Map the parts to the correct columns based on filename structure:
    # depth_project_date_time_location_replicate
    data['depth'] = filename_parts[0]
    data['project'] = filename_parts[1]
    data['date'] = filename_parts[2]
    data['time'] = filename_parts[3]
    data['replicate'] = filename_parts[5]  # location is at index 4, replicate at 5
    
    # Add treatment column if requested
    if treatment:
        data['treatment'] = data['replicate'].str[0]
    
    # Correct spelling mistake
    data['project'] = data['project'].str.replace('WinterSamplingCampagne', 'WinterSamplingCampaign')
    
    # Replace "," for "." in depth and convert to numeric
    data['depth'] = data['depth'].str.replace(',', '.')
    data['depth'] = pd.to_numeric(data['depth'])
    
    # Get only filename (without extension) and rename to Filename
    data['Filename'] = data[column].apply(lambda x: Path(x).stem)
    
    return data

def find_files(folder: Path, subfolder_name: str, extension: str):
    """Finds all files with a given extension in a specific subfolder."""
    found_files = []
    for subfolder in folder.rglob(subfolder_name):
        if subfolder.is_dir():
            found_files.extend(subfolder.rglob(f"*.{extension}"))
    return found_files

def restructure_data(classification_files: List[Path], vignette_files: List[Path], output_path: Path):
    matched_pairs = match_files_by_lcs(classification_files, vignette_files)
    print(f"Found {len(matched_pairs)} file pairs to process.")

    for c_file, v_file in matched_pairs:
        # try:
        # 1. Read and structure classification data
        df = pd.read_csv(c_file)
        
        # Apply the data restructuring function (equivalent to R's DataRestructuring)
        df = data_restructuring(df, 'path')
        
        # Select and order columns as in the R script
        df = df[['Filename', 'project', 'date', 'time', 'replicate', 'depth', 'prediction', 'label']]

        # 4. Read measurement file, using a semicolon separator
        measurement_df = pd.read_csv(v_file, sep=';')

        # 5. Merge dataframes
        merged_df = pd.merge(df, measurement_df, on='Filename', how='left')
        
        # 6. Relocate Filename to the end
        filename_col = merged_df.pop('Filename')
        merged_df['Filename'] = filename_col
        
        save_path = output_path / df['project'].iloc[0] / 'object_data' / df['date'].iloc[0] / df['time'].iloc[0].lower()
        save_path.mkdir(parents=True, exist_ok=True)
        
        file_path = save_path / f"{df['replicate'].iloc[0]}_{df['project'].iloc[0]}_{df['date'].iloc[0]}_{df['time'].iloc[0].lower()}_object_data.csv"
        
        # 8. Write to CSV
        merged_df.to_csv(file_path, index=False)
        print(f"Successfully processed and saved data to {file_path}")

def wrapper(input_path: Path):
    if not input_path.is_dir():
        print(f"Error: Input folder not found at {input_path}")
        return
    
    output_path = Path(input_path) / 'restructured'
    output_path.mkdir(parents=True, exist_ok=True)

    classification_files = find_files(input_path, "classification", "csv")
    vignette_files = find_files(input_path, "vignettes", "csv")

    restructure_data(classification_files, vignette_files, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Restructure data files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input directory path')
    args = parser.parse_args()
    
    wrapper(Path(args.input)) 