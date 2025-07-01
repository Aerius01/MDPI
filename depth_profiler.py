import os
import pandas as pd
import shutil
from datetime import datetime
from constants import BASE_FILENAME_PATTERN, TIMESTEP, CONSTANTS
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ProfileConfig:
    """Configuration for depth profiling."""
    csv_separator: str = CONSTANTS.CSV_SEPARATOR
    csv_header_row: int = CONSTANTS.CSV_HEADER_ROW
    csv_columns: List[int] = None
    csv_skipfooter: int = CONSTANTS.CSV_SKIPFOOTER
    depth_multiplier: float = CONSTANTS.DEPTH_MULTIPLIER
    
    def __post_init__(self):
        if self.csv_columns is None:
            self.csv_columns = [0, 1]

class MetadataExtractor:
    """Handles metadata extraction from image paths."""
    
    @staticmethod
    def extract_from_path(image_path: str) -> Tuple[str, str, str, str, str]:
        """Extract metadata from image path."""
        path_parts = image_path.split(os.path.sep)
        return path_parts[-5], path_parts[-4], path_parts[-3], path_parts[-2], path_parts[-1]
    
    @staticmethod
    def parse_datetime_from_filename(filename: str) -> datetime:
        """Parse datetime from filename using regex."""
        match = BASE_FILENAME_PATTERN.search(filename)
        if not match:
            raise ValueError(f"Could not parse datetime from filename: {filename}")
        return pd.to_datetime(datetime.strptime(match.group(1), '%Y%m%d_%H%M%S%f'))

class CSVProcessor:
    """Handles CSV file processing for depth data."""
    
    def __init__(self, config: ProfileConfig):
        self.config = config
    
    def load_and_process(self, csv_path: str) -> pd.DataFrame:
        """Load and process CSV depth data."""
        csv_data = pd.read_csv(
            csv_path, 
            sep=self.config.csv_separator, 
            header=self.config.csv_header_row,
            usecols=self.config.csv_columns,
            names=['time', 'depth'], 
            index_col='time',
            skipfooter=self.config.csv_skipfooter, 
            engine='python'
        )
        
        # Process timestamps and depths
        csv_data.index = pd.to_datetime(csv_data.index, format='%d.%m.%Y %H:%M:%S.%f')
        csv_data['depth'] = csv_data['depth'].str.replace(',', '.').astype(float) * self.config.depth_multiplier
        
        return csv_data

class DepthProfiler:
    """Main class for depth profiling operations."""
    
    def __init__(self, config: ProfileConfig = None):
        self.config = config or ProfileConfig()
        self.metadata_extractor = MetadataExtractor()
        self.csv_processor = CSVProcessor(self.config)
    
    def _find_csv_file(self, directory: str) -> str:
        """Find CSV file in directory."""
        csv_files = [f for f in os.listdir(directory) if f.endswith(CONSTANTS.CSV_EXTENSION)]
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {directory}")
        return os.path.join(directory, csv_files[0])
    
    def _generate_output_filenames(self, depths: List[float], project: str, date: str, time: str, location: str, output_dir: str) -> List[str]:
        """Generate output filenames from depth values and metadata."""
        filename_parts = [project, date, time, location]
        clean_parts = [part.replace("_", "-") for part in filename_parts]
        
        return [
            os.path.join(output_dir, f'{depth:.3f}_{"_".join(clean_parts)}{CONSTANTS.TIFF_EXTENSION}')
            for depth in depths
        ]
    
    def process_group(self, image_group: List[str], output_path: str) -> List[str]:
        """Process image group for depth profiling."""
        print(f"[PROFILING]: Starting depth matching...")
        
        # Extract metadata
        project, date, time, location, filename = self.metadata_extractor.extract_from_path(image_group[0])
        root_dir = os.path.dirname(image_group[0])
        
        print(f"[PROFILING]: Processing group: {project}/{date}/{time}/{location} ({len(image_group)} images)")
        
        # Setup output directory
        output_dir = os.path.join(output_path, project, date, time, location)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process CSV data
        csv_path = self._find_csv_file(root_dir)
        csv_data = self.csv_processor.load_and_process(csv_path)
        
        # Calculate timestamps and depths
        image_time = self.metadata_extractor.parse_datetime_from_filename(filename)
        timestamps = [image_time + (i * TIMESTEP) for i in range(len(image_group))]
        nearest_indices = csv_data.index.get_indexer(timestamps, method='nearest')
        depth_values = csv_data.iloc[nearest_indices]['depth'].values
        
        # Generate output filenames and copy files
        print(f"[PROFILING]: Calculating timestamps and depth values...")
        output_filenames = self._generate_output_filenames(depth_values, project, date, time, location, output_dir)
        
        print(f"[PROFILING]: Saving files...")
        for source, dest in zip(image_group, output_filenames):
            shutil.copy2(source, dest)
        
        print(f"[PROFILING]: Depth matching completed successfully!")
        return sorted(output_filenames)