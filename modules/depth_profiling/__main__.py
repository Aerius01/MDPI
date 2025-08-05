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
from .config import ProfileConfig
from .profiler import DepthProfiler

def main():
    parser = argparse.ArgumentParser(
        description='Process images for depth profiling using CSV depth data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m modules.depth_profiling -i ./raw_images -c 2.4
  python -m modules.depth_profiling -i ./raw_images -o ./output/depth_profiles -c 2.4 --depth-multiplier 5.0
  python -m modules.depth_profiling -i ./raw_images -o ./output/depth_profiles -c 2.4 --csv-separator ","
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input directory containing images and CSV depth data')
    parser.add_argument('-o', '--output', default='./output/depth_profiles',
                        help='Top-level output directory for the image-depth mapping .csv ')
    parser.add_argument('-c', '--capture-rate', type=float, required=True,
                        help='The image capture rate in hertz (Hz) of the MDPI')
    parser.add_argument('--csv-separator', default=';',
                        help='CSV separator character (default: ;)')
    parser.add_argument('--csv-header-row', type=int, default=6,
                        help='CSV header row index (default: 6)')
    parser.add_argument('--csv-skipfooter', type=int, default=1,
                        help='Number of CSV footer rows to skip (default: 1)')
    parser.add_argument('--depth-multiplier', type=float, default=10.0,
                        help='Depth multiplier for CSV values (default: 10.0)')
    parser.add_argument('--csv-columns', nargs=2, type=int, default=[0, 1],
                        metavar=('TIME_COL', 'DEPTH_COL'),
                        help='CSV column indices for time and depth (default: 0 1)')
    
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
        
        metadata = {
            "project": path_metadata["project"],
            "date_str": path_metadata["date_str"],
            "cycle": path_metadata["cycle"],
            "location": path_metadata["location"],
            "time": recording_start_time.strftime("%H%M%S")
        }
        
        print(f"[PROFILING]: Processing group: {metadata['project']}/{metadata['date']}/{metadata['cycle']}/{metadata['location']}")
        
        # Configure depth profiler
        config = ProfileConfig(
            capture_rate=args.capture_rate,
            csv_separator=args.csv_separator,
            csv_header_row=args.csv_header_row,
            csv_columns=args.csv_columns,
            csv_skipfooter=args.csv_skipfooter,
            depth_multiplier=args.depth_multiplier
        )
        
        # Run depth profiling
        profiler = DepthProfiler(config)
        df = profiler.process_depth_data(image_paths, pressure_sensor_csv_path, recording_start_time)
        
        if df is not None:
            output_file = save_csv_data(df, metadata, output_path, "depth_profiling_output")
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