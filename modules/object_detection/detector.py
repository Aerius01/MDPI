import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import List, Tuple
import os
from .config import DetectionConfig
from .overlap_corrector import OverlapCorrector
from .region_processor import RegionProcessor

class ObjectDetector:
    """Main class for object detection operations."""
    
    def __init__(self, config: DetectionConfig = None):
        self.config = config or DetectionConfig()
        self.overlap_corrector = OverlapCorrector(self.config)
        self.region_processor = RegionProcessor(self.config)
    
    def _parse_image_metadata(self, image_path: str) -> Tuple[str, str, str, str]:
        """Extract metadata from image filename."""
        filename = Path(image_path).stem
        _, project, date, time, location = filename.split('_')
        return project, date, time, location
    
    def _load_and_threshold_images(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Load images and apply thresholding."""
        images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
        img_bins = [cv2.threshold(img, self.config.threshold_value, 
                                 self.config.threshold_max, cv2.THRESH_BINARY_INV)[1] 
                   for img in images]
        return images, img_bins
    
    def _apply_size_filtering(self, label_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Apply size-based filtering to labeled images."""
        return [
            remove_small_objects(label_img > 0, min_size=self.config.min_object_size) & 
            ~remove_small_objects(label_img > 0, min_size=self.config.max_object_size)
            for label_img in label_imgs
        ]
    
    def _process_batch(self, batch_corrected_bins: List[np.ndarray], batch_images: List[np.ndarray], 
                      batch_names: List[str], group_output_path: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Process a batch of corrected binary images to detect and analyze objects."""
        # Perform object detection and size filtering
        batch_label_imgs = [label(img) for img in batch_corrected_bins]
        batch_filtered_imgs = self._apply_size_filtering(batch_label_imgs)
        
        # Extract regions and process each image
        batch_regions = [regionprops(label(img), image) 
                        for img, image in zip(batch_filtered_imgs, batch_images)]
        
        batch_data_dfs = []
        batch_output_paths = []
        for region, image, img_name in zip(batch_regions, batch_images, batch_names):
            data_df, output_paths = self.region_processor.process_regions(region, image, img_name, group_output_path)
            if not data_df.empty:
                batch_data_dfs.append(data_df)
            batch_output_paths.extend(output_paths)
        
        return batch_data_dfs, batch_output_paths
    
    def process_group(self, image_group: List[str], output_path: str) -> List[str]:
        """Main function to detect objects in a given image group."""
        print(f"[DETECTION]: Starting object detection...")
        
        # Extract names and metadata
        img_names = [Path(path).stem for path in image_group]
        project, date, time, location = self._parse_image_metadata(image_group[0])
        
        print(f"[DETECTION]: Processing group: {project}/{date}/{time}/{location} ({len(image_group)} images)")
        
        # Create output directory
        group_output_path = os.path.join(output_path, project, date, time, location)
        os.makedirs(group_output_path, exist_ok=True)
        
        # Step 1: Load and threshold images
        print(f"[DETECTION]: Loading and thresholding...")
        images, img_bins = self._load_and_threshold_images(image_group)
        
        # Step 2: Apply overlap correction
        print(f"[DETECTION]: Applying depth overlap correction...")
        corrected_img_bins = self.overlap_corrector.apply_corrections(img_bins, img_names)
        
        # Step 3: Process in batches
        num_batches = int(np.ceil(len(image_group) / self.config.batch_size))
        print(f"[DETECTION]: Performing object detection in {num_batches} batches...")
        
        all_data_dfs = []
        all_output_paths = []
        for j in tqdm(range(0, len(image_group), self.config.batch_size), desc='[DETECTION]'):
            batch_end = j + self.config.batch_size
            batch_corrected_bins = corrected_img_bins[j:batch_end]
            batch_images = images[j:batch_end]
            batch_names = img_names[j:batch_end]
            
            batch_data_dfs, batch_output_paths = self._process_batch(batch_corrected_bins, batch_images, 
                                                                   batch_names, str(group_output_path))
            all_data_dfs.extend(batch_data_dfs)
            all_output_paths.extend(batch_output_paths)
        
        # Save results
        if all_data_dfs:
            combined_df = pd.concat(all_data_dfs, ignore_index=True)
            output_file = os.path.join(group_output_path, f'objectMeasurements_{project}_{date}_{time}_{location}{self.config.csv_extension}')
            combined_df.to_csv(output_file, sep=self.config.csv_separator, index=False)
            print(f"[DETECTION]: Detection completed successfully! Total objects detected: {len(combined_df)}")
        else:
            print(f"[DETECTION]: Detection completed. No objects detected.")
        
        # Return sorted list of output image file paths
        return sorted(all_output_paths) 