"""
Common utilities for command line interfaces across processing modules.
"""

import os
from imutils import paths
from typing import List, Callable, Optional

class CommonCLI:
    """Common command line interface utilities for image processing modules."""
    @staticmethod
    def get_image_group_from_folder(image_folder: str, sort_key: Optional[Callable] = None) -> List[str]:
        """Get list of image paths from a folder with custom sorting."""
        if not os.path.exists(image_folder):
            raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
        
        try:
            image_group = list(paths.list_images(image_folder))
            if not image_group:
                raise ValueError(f"No images found in folder: {image_folder}")
            
            if sort_key:
                image_group = sorted(image_group, key=sort_key)
            else:
                image_group = sorted(image_group)
                
            return image_group
        except Exception as e:
            raise ValueError(f"Error reading images from {image_folder}: {str(e)}")
    
    @staticmethod
    def validate_output_path(output_path: str) -> str:
        """Validate and create output path if needed."""
        try:
            os.makedirs(output_path, exist_ok=True)
            return output_path
        except Exception as e:
            raise ValueError(f"Cannot create output path {output_path}: {str(e)}") 