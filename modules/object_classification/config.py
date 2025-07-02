from dataclasses import dataclass
from typing import List

@dataclass
class ClassificationConfig:
    """Configuration for the classification pipeline."""
    image_group: List[str]  # List of image file paths
    model_path: str
    output_path: str = './classification'
    batch_size: int = 32
    input_size: int = 50
    input_depth: int = 1
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ['cladocera', 'copepod', 'junk', 'rotifer'] 