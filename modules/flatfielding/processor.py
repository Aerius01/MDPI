import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import List
from .config import FlatfieldConfig

class FlatfieldProcessor:
    """Handles flatfielding operations on image groups."""
    
    def __init__(self, config: FlatfieldConfig = None):
        self.config = config or FlatfieldConfig()
    
    def _extract_metadata_from_filename(self, filename: str) -> tuple:
        """Extract metadata from filename: depth_project_date_time_location.tiff"""
        base_name = os.path.splitext(filename)[0]
        _, project, date, time, location = base_name.split('_')
        return project, date, time, location
    
    def _calculate_average_image(self, image_paths: List[str]) -> np.ndarray:
        """Calculate the average image for flatfielding."""
        print(f"[FLATFIELDING]: Calculating average image...")
        images = np.array([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_paths])
        return np.average(images, axis=0).astype('uint8'), images
    
    def _process_batch(self, batch_images: np.ndarray, batch_names: List[str], 
                      flatfield_image: np.ndarray, output_dir: str) -> List[str]:
        """Process a batch of images for flatfielding."""
        # Perform flatfielding on batch (divide and clip operations)
        flatfielded_batch = np.divide(batch_images, flatfield_image) * self.config.normalization_factor
        flatfielded_batch = np.clip(flatfielded_batch, 0, 255).astype('uint8')
        
        # Save batch of flatfielded images
        output_paths = []
        for img, name in zip(flatfielded_batch, batch_names):
            output_file_path = os.path.join(output_dir, f'{name}{self.config.output_format}')
            cv2.imwrite(output_file_path, img)
            output_paths.append(output_file_path)
        
        return output_paths
    
    def process_group(self, image_group: List[str], output_path: str) -> List[str]:
        """Process image group for flatfielding."""
        print(f"[FLATFIELDING]: Starting flatfielding...")
        
        # Get image names without extension for saving later
        img_names = [os.path.splitext(os.path.basename(path))[0] for path in image_group]
        
        # Extract metadata from filename
        filename = os.path.basename(image_group[0])
        project, date, time, location = self._extract_metadata_from_filename(filename)
        
        print(f"[FLATFIELDING]: Processing group: {project}/{date}/{time}/{location} ({len(image_group)} images)")
        
        # Create output path
        output_dir = os.path.join(output_path, project, date, time, location)
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate the average image and load all images
        flatfield_image, images = self._calculate_average_image(image_group)
        
        # Process images in batches
        all_output_paths = []
        num_batches = int(np.ceil(len(image_group) / self.config.batch_size))
        print(f"[FLATFIELDING]: Flatfielding images in {num_batches} batches...")
        
        for j in tqdm(range(0, len(image_group), self.config.batch_size), desc='[FLATFIELDING]'):
            batch_end = j + self.config.batch_size
            batch_images = images[j:batch_end]
            batch_names = img_names[j:batch_end]
            
            batch_output_paths = self._process_batch(batch_images, batch_names, flatfield_image, output_dir)
            all_output_paths.extend(batch_output_paths)
        
        print(f"[FLATFIELDING]: Flatfielding completed successfully!")
        return sorted(all_output_paths) 