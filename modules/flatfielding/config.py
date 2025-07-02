from dataclasses import dataclass
from modules.common.constants import CONSTANTS

@dataclass
class FlatfieldConfig:
    """Configuration for flatfielding."""
    batch_size: int = CONSTANTS.BATCH_SIZE
    normalization_factor: int = CONSTANTS.NORMALIZATION_FACTOR
    output_format: str = CONSTANTS.JPEG_EXTENSION 