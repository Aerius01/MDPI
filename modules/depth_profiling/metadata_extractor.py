import os
import pandas as pd
from datetime import datetime
from modules.common.constants import BASE_FILENAME_PATTERN
from typing import Tuple

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