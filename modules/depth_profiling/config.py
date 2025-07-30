from dataclasses import dataclass
from modules.common.constants import CONSTANTS
from typing import List

@dataclass
class ProfileConfig:
    """Configuration for depth profiling."""
    capture_rate: float
    csv_separator: str = CONSTANTS.CSV_SEPARATOR
    csv_header_row: int = CONSTANTS.CSV_HEADER_ROW
    csv_columns: List[int] = None
    csv_skipfooter: int = CONSTANTS.CSV_SKIPFOOTER
    depth_multiplier: float = CONSTANTS.DEPTH_MULTIPLIER
    
    def __post_init__(self):
        if self.csv_columns is None:
            self.csv_columns = [0, 1] 