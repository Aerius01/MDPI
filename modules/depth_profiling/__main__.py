#!/usr/bin/env python3
"""
Command line interface for depth profiling module.
Usage: python -m modules.depth_profiling [options]
"""
import argparse
import sys
from pathlib import Path
from modules.common.cli_utils import CommonCLI
from modules.common.parser import parse_path_metadata, parse_file_metadata, find_single_csv_file
from modules.common.file_utils import save_csv_data
from .profiler import DepthProfiler
import datetime

def main():
    parser = argparse.ArgumentParser(
        description='Process images for depth profiling using CSV depth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.depth_profiling -i ./raw_images -c 2.4
  python -m modules.depth_profiling -i ./raw_images -o ./output -c 2.4
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing raw MDPI images and the pressure sensor (depth) data .csv file')
    parser.add_argument('-o', '--output', default='./output',
                        help='The root output directory for the image-depth mapping csv file. The full path will be <output_directory>/project/date/cycle/location/depth_profiles.csv')
    parser.add_argument('-c', '--capture-rate', type=float, required=True,
                        help='The image capture rate in hertz (Hz) of the MDPI')
    
    args = parser.parse_args()
    
    try:
        # Validate and create output path
        output_path = CommonCLI.validate_output_path(args.output)
        
        # Get image paths from input directory
        input_path = Path(args.input)
        path_metadata = parse_path_metadata(input_path)
        file_metadata = parse_file_metadata(input_path, path_metadata["recording_start_date"])
        pressure_sensor_csv_path = find_single_csv_file(input_path)
        
        image_paths = file_metadata["raw_imgs"]
        print(f"[PROFILING]: Found {len(image_paths)} images")
        
        # Extract metadata for processing and saving
        recording_start_time = file_metadata["recording_start_time"]
        recording_start_date = path_metadata["recording_start_date"]
        start_datetime = datetime.datetime.combine(recording_start_date, recording_start_time)

        metadata = {
            "project": path_metadata["project"],
            "date_str": path_metadata["date_str"],
            "cycle": path_metadata["cycle"],
            "location": path_metadata["location"],
            "time": recording_start_time.strftime("%H%M%S")
        }
        
        print(f"[PROFILING]: Processing group: {metadata['project']}/{metadata['date_str']}/{metadata['cycle']}/{metadata['location']}")
        
        # Run depth profiling
        profiler = DepthProfiler()
        df = profiler.map_images_to_depths(image_paths, pressure_sensor_csv_path, start_datetime, args.capture_rate)
        
        if df is not None:
            output_file = save_csv_data(df, metadata, output_path, "depth_profiles")
            if output_file:
                print(f"[PROFILING]: Processing completed successfully!")
                print(f"[PROFILING]: 1 file saved to {output_file}")
            else:
                raise Exception("Failed to save CSV file.")
        else:
            raise Exception("Failed to process depth data.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 