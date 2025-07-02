import pandas as pd
from .config import ProfileConfig

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