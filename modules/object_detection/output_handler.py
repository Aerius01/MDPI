import pandas as pd
from pathlib import Path
import os
from .detection_data import DetectionData
from dataclasses import dataclass

@dataclass
class OutputHandler:
    csv_extension: str
    csv_separator: str

    def create_dataframe(self, data_list: list, detection_data: DetectionData) -> pd.DataFrame:
        # Create a combined DataFrame from all processed regions
        if not data_list:
            return pd.DataFrame()
        
        # Create dataframe
        combined_df = pd.DataFrame(data_list)

        # Join with depth profiles
        combined_df = pd.merge(combined_df, detection_data.depth_profiles_df, on='image_id', how='left')

        # Add metadata
        combined_df['project'] = detection_data.project
        combined_df['recording_start_date'] = detection_data.recording_start_date
        combined_df['cycle'] = detection_data.cycle
        combined_df['location'] = detection_data.location
        
        # Sort the dataframe by image_id and then by replicate
        combined_df = combined_df.sort_values(by=['image_id', 'replicate'])
        
        # Reorder columns to have FileName first, then metadata, then other data
        cols = ['FileName', 'project', 'recording_start_date', 'cycle', 'location', 'image_id', 'replicate', 'depth'] + \
               [col for col in combined_df.columns if col not in ['FileName', 'project', 'recording_start_date', 'cycle', 'location', 'image_id', 'replicate', 'depth']]
        combined_df = combined_df[cols]

        return combined_df

    def save_dataframe(self, combined_df: pd.DataFrame, output_path: str) -> None:
        """Save detection results to CSV and text files."""

        if combined_df.empty:
            print(f"[DETECTION]: Detection completed. No objects detected.")
            return

        # Save results
        print(f"[DETECTION]: Detection completed successfully! Total objects detected: {len(combined_df)}")

        csv_output_file = os.path.join(Path(output_path).parent, f'object_data{self.csv_extension}')
        combined_df.to_csv(csv_output_file, sep=self.csv_separator, index=False)
        print(f"[DETECTION]: Saved results to {csv_output_file}") 