import argparse
import os
from pathlib import Path
import pandas as pd
import re

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
    # 0     1      2     3     4        5
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

def longest_common_substring(s1: str, s2: str) -> str:
    """Finds the longest common substring between two strings."""
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def find_files(folder: Path, subfolder_name: str, extension: str):
    """Finds all files with a given extension in a specific subfolder."""
    found_files = []
    for subfolder in folder.rglob(subfolder_name):
        if subfolder.is_dir():
            found_files.extend(subfolder.rglob(f"*.{extension}"))
    return found_files

def main():
    """Main function to find and print file paths."""
    parser = argparse.ArgumentParser(description="Restructure data by finding classification and vignette files.")
    parser.add_argument("-i", "--input_folder", type=Path, required=True, help="Input folder to search for data.")
    parser.add_argument("-o", "--output_folder", type=Path, required=True, help="Output folder to save restructured data.")
    args = parser.parse_args()

    input_path = args.input_folder
    output_path = args.output_folder
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.is_dir():
        print(f"Error: Input folder not found at {input_path}")
        return

    classification_files = find_files(input_path, "classification", "csv")
    vignette_files = find_files(input_path, "vignettes", "csv")

    # Determine the smaller list to be the "root" for matching
    root_files = classification_files
    other_files = vignette_files
    # Swap if vignettes are fewer, to iterate over the smaller list
    if len(vignette_files) < len(classification_files):
        root_files, other_files = other_files, root_files

    potential_matches = []
    for root_file in root_files:
        best_match = None
        max_lcs = -1
        for other_file in other_files:
            lcs = longest_common_substring(root_file.stem, other_file.stem)
            if len(lcs) > max_lcs:
                max_lcs = len(lcs)
                best_match = other_file
        
        if best_match:
            # The order of the pair in the tuple depends on which list was the root
            if len(vignette_files) < len(classification_files):
                 # root is vignettes, so (v_file, c_file) -> store as (c_file, v_file)
                potential_matches.append((max_lcs, best_match, root_file))
            else:
                # root is classifications, so (c_file, v_file)
                potential_matches.append((max_lcs, root_file, best_match))

    # Sort potential matches by LCS score, descending
    potential_matches.sort(key=lambda x: x[0], reverse=True)

    matched_pairs = []
    used_c_files = set()
    used_v_files = set()

    for _, c_file, v_file in potential_matches:
        if c_file not in used_c_files and v_file not in used_v_files:
            matched_pairs.append((c_file, v_file))
            used_c_files.add(c_file)
            used_v_files.add(v_file)

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

        # except Exception as e:
        #     print(f"Error processing pair ({c_file}, {v_file}): {e}")


if __name__ == "__main__":
    main() 