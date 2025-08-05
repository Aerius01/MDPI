import argparse
import sys
import os
from pipeline_profile import Profile
from modules.duplicate_detection import DuplicateDetector, DuplicateConfig
from modules.depth_profiling.profiler import DepthProfiler
from modules.common.file_utils import save_csv_data
from datetime import datetime
from modules.common.constants import CONSTANTS

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
        removed_paths = detector.process_images(profile.raw_imgs)
        profile.set_deduplicated_imgs(removed_paths)
        
        # Configure and run depth profiling
        print("Running depth profiling...")
        depth_profiler = DepthProfiler()
        
        # Combine date and time for a full datetime object
        start_datetime = datetime.combine(profile.recording_start_date, profile.recording_start_time)
        
        depth_mapping_df = depth_profiler.map_images_to_depths(
            image_paths=profile.deduplicated_imgs,
            pressure_sensor_csv_path=profile.pressure_sensor_csv_path,
            recording_start_datetime=start_datetime,
            capture_rate=args.capture_rate
        )

        profile.set_depth_mapping_df(depth_mapping_df)
        
        # Save depth data to CSV
        print("Saving depth data to CSV...")
        metadata = {
            "project": profile.project,
            "date_str": profile.recording_start_date.strftime('%Y%m%d'),
            "cycle": profile.cycle,
            "location": profile.location
        }
        csv_path = save_csv_data(depth_mapping_df, metadata, profile.output_folder, "depth_profiles")
        
        print("Processing complete.")
        
        # Verify the number of removed files
        assert len(removed_paths) == len(profile.raw_imgs) - len(profile.deduplicated_imgs), \
            "The number of removed files does not match the difference in file counts."
            
        # Verify that the deduplicated_imgs list is correctly updated
        if len(removed_paths) > 0:
            assert profile.deduplicated_imgs, "Deduplicated images list should not be empty."
        
        # Verify depth profiling results
        assert depth_mapping_df is not None, "Depth profiling failed, DataFrame is None."
        assert not depth_mapping_df.empty, "Depth profiling returned an empty DataFrame."
        assert len(depth_mapping_df) == len(profile.deduplicated_imgs), \
            "The number of rows in the depth DataFrame does not match the number of deduplicated images."
        
        # Verify CSV saving
        assert csv_path is not None, "CSV saving failed, path is None."
        assert os.path.exists(csv_path), f"CSV file was not created at {csv_path}."
        
        print(f"Deduplicated images count: {len(profile.deduplicated_imgs)}")
        print(f"Initial images count: {len(profile.raw_imgs)}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
