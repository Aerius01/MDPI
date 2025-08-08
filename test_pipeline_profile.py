import argparse
import sys
import os
from pipeline_profile import Profile
from modules.duplicate_detection import DuplicateDetector, DuplicateConfig
from modules.depth_profiling.profiler import DepthProfiler
from datetime import datetime
from modules.common.constants import CONSTANTS


# Destructured CONSTANTS for cleaner readability
CSV_EXTENSION = CONSTANTS.CSV_EXTENSION
CSV_SEPARATOR = CONSTANTS.CSV_SEPARATOR
CSV_HEADER_ROW = CONSTANTS.CSV_HEADER_ROW
CSV_COLUMNS = CONSTANTS.CSV_COLUMNS
CSV_SKIPFOOTER = CONSTANTS.CSV_SKIPFOOTER
DEPTH_MULTIPLIER = CONSTANTS.DEPTH_MULTIPLIER
TIME_COLUMN_NAME = "time"
DEPTH_COLUMN_NAME = "depth"

# Destructured CONSTANTS for cleaner readability
BATCH_SIZE = CONSTANTS.BATCH_SIZE
NORMALIZATION_FACTOR = CONSTANTS.NORMALIZATION_FACTOR
IMAGE_EXTENSION = CONSTANTS.JPEG_EXTENSION

def main():
    """
    A command-line script to test the Profile class from pipeline_profile.py.
    """
    parser = argparse.ArgumentParser(
        description='Test the Profile class and duplicate detection.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('-d', '--directory', required=True,
                        help='Input directory for the profile.')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for the profile.')
    parser.add_argument('-c', '--capture-rate', type=float, default=2.4,
                        help='The image capture rate in hertz (Hz).')
    
    args = parser.parse_args()
    
    try:
        # Create a Profile instance
        print(f"Creating profile for directory: {args.directory}")
        profile = Profile(args.directory, args.output)
        
        # Configure and run duplicate detection
        print("Running duplicate detection...")
        dup_config = DuplicateConfig(remove=True)
        detector = DuplicateDetector(dup_config)
        removed_paths = detector.process_images(profile.raw_img_paths)
        profile.set_deduplicated_imgs(removed_paths)
        
        # Configure and run depth profiling
        print("Running depth profiling...")
        depth_profiler = DepthProfiler(
            csv_separator=CSV_SEPARATOR,
            csv_header_row=CSV_HEADER_ROW,
            csv_columns=CSV_COLUMNS,
            csv_skipfooter=CSV_SKIPFOOTER,
            depth_multiplier=DEPTH_MULTIPLIER,
            time_column_name=TIME_COLUMN_NAME,
            depth_column_name=DEPTH_COLUMN_NAME
        )
        
        # Combine date and time for a full datetime object
        start_datetime = datetime.combine(profile.recording_start_date, profile.recording_start_time)

        depth_mapping_df = depth_profiler.map_images_to_depths(profile.deduplicated_imgs, profile.pressure_sensor_csv_path, start_datetime, args.capture_rate)
        profile.set_depth_mapping_df(depth_mapping_df)

        if depth_mapping_df is not None:
            output_csv_path = os.path.join(profile.output_path, "depth_profiles" + CSV_EXTENSION)
            depth_mapping_df.to_csv(output_csv_path, index=False)
            print(f"[PROFILING]: Successfully saved data to {output_csv_path}")
            print(f"[PROFILING]: Processing completed successfully!")
        else:
            raise Exception("Failed to process depth data.")
        
        print("Processing complete.")
        
        # Verify the number of removed files
        assert len(removed_paths) == len(profile.raw_img_paths) - len(profile.deduplicated_imgs), \
            "The number of removed files does not match the difference in file counts."
            
        # Verify that the deduplicated_imgs list is correctly updated
        if len(removed_paths) > 0:
            assert profile.deduplicated_imgs, "Deduplicated images list should not be empty."
        
        # Verify depth profiling results
        assert depth_mapping_df is not None, "Depth profiling failed, DataFrame is None."
        assert not depth_mapping_df.empty, "Depth profiling returned an empty DataFrame."
        assert len(depth_mapping_df) == len(profile.deduplicated_imgs), \
            "The number of rows in the depth DataFrame does not match the number of deduplicated images."
        
        print(f"Deduplicated images count: {len(profile.deduplicated_imgs)}")
        print(f"Initial images count: {len(profile.raw_img_paths)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
