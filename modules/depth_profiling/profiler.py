import os
import shutil
from modules.common.constants import TIMESTEP, CONSTANTS
from typing import List
from .config import ProfileConfig
from .metadata_extractor import MetadataExtractor
from .csv_processor import CSVProcessor

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