import cv2
import pandas as pd
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
import numpy as np
from pathlib import Path
from constants import CONSTANTS
from typing import List, Tuple
from dataclasses import dataclass
import os
import argparse
from cli_utils import CommonCLI

@dataclass
class DetectionConfig:
    """Configuration for object detection."""
    # Depth overlap correction
    depth_multiplier: int = CONSTANTS.DEPTH_MULTIPLIER_CM
    image_height_cm: float = CONSTANTS.IMAGE_HEIGHT_CM
    image_height_pixels: int = CONSTANTS.IMAGE_HEIGHT_PIXELS
    
    # Image thresholding
    threshold_value: int = CONSTANTS.THRESHOLD_VALUE
    threshold_max: int = CONSTANTS.THRESHOLD_MAX
    
    # Object filtering
    min_object_size: int = CONSTANTS.MIN_OBJECT_SIZE
    max_object_size: int = CONSTANTS.MAX_OBJECT_SIZE
    
    # Region filtering
    max_eccentricity: float = CONSTANTS.MAX_ECCENTRICITY
    max_mean_intensity: int = CONSTANTS.MAX_MEAN_INTENSITY
    min_major_axis_length: int = CONSTANTS.MIN_MAJOR_AXIS_LENGTH
    max_min_intensity: int = CONSTANTS.MAX_MIN_INTENSITY
    
    # Object cropping
    small_object_padding: int = CONSTANTS.SMALL_OBJECT_PADDING
    medium_object_padding: int = CONSTANTS.MEDIUM_OBJECT_PADDING
    large_object_padding: int = CONSTANTS.LARGE_OBJECT_PADDING
    small_object_threshold: int = CONSTANTS.SMALL_OBJECT_THRESHOLD
    medium_object_threshold: int = CONSTANTS.MEDIUM_OBJECT_THRESHOLD
    
    # File operations
    csv_separator: str = CONSTANTS.CSV_SEPARATOR
    jpeg_extension: str = CONSTANTS.JPEG_EXTENSION
    csv_extension: str = CONSTANTS.CSV_EXTENSION
    batch_size: int = CONSTANTS.BATCH_SIZE

# Column definitions for consistent DataFrame creation
MEASUREMENT_COLUMNS = [
    'Filename', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
    'Orientation', 'EquivDiameter', 'Solidity', 'Extent', 'MaxIntensity', 
    'MeanIntensity', 'MinIntensity', 'Perimeter'
]

class OverlapCorrector:
    """Handles depth overlap correction calculations."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def calculate_corrections(self, img_names: List[str]) -> np.ndarray:
        """Calculate pixel overlaps for depth correction."""
        depths = np.array([float(name.split('_')[0]) for name in img_names])
        image_bottom_depths = depths * self.config.depth_multiplier + self.config.image_height_cm
        image_top_depths = depths * self.config.depth_multiplier
        
        overlaps = np.zeros(len(depths))
        overlaps[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
        
        return np.round((overlaps / self.config.image_height_cm) * self.config.image_height_pixels).astype(int)
    
    def apply_corrections(self, img_bins: List[np.ndarray], img_names: List[str]) -> List[np.ndarray]:
        """Apply depth overlap correction to binary images."""
        pixel_overlaps = self.calculate_corrections(img_names)
        return [
            np.where(np.arange(img_bin.shape[0])[:, None] < pixel_overlap, 
                    self.config.threshold_max, img_bin)
            for img_bin, pixel_overlap in zip(img_bins, pixel_overlaps)
        ]

class RegionProcessor:
    """Handles region filtering and processing."""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def filter_regions(self, regions: List) -> List:
        """Apply all filtering criteria to regions and return valid ones."""
        if not regions:
            return []
        
        return [
            r for r in regions
            if (r.eccentricity < self.config.max_eccentricity and
                r.mean_intensity < self.config.max_mean_intensity and
                r.major_axis_length > self.config.min_major_axis_length and
                r.min_intensity < self.config.max_min_intensity)
        ]
    
    def calculate_crop_padding(self, major_axis_length: float) -> int:
        """Calculate appropriate padding based on object size."""
        if major_axis_length < self.config.small_object_threshold:
            return self.config.small_object_padding
        elif major_axis_length < self.config.medium_object_threshold:
            return self.config.medium_object_padding
        else:
            return self.config.large_object_padding
    
    def extract_and_save_region(self, region, image: np.ndarray, img_name: str, 
                               index: int, output_path: str) -> Tuple[List, str]:
        """Extract region data and save cropped image."""
        # Extract region measurements
        region_data = [
            f"{img_name}_{index}",
            region.area,
            region.major_axis_length,
            region.minor_axis_length,
            region.eccentricity,
            region.orientation,
            region.equivalent_diameter,
            region.solidity,
            region.extent,
            region.max_intensity,
            region.mean_intensity,
            region.min_intensity,
            region.perimeter
        ]
        
        # Calculate crop boundaries
        row, col = int(region.centroid[0]), int(region.centroid[1])
        padding = self.calculate_crop_padding(region.major_axis_length)
        
        minr = max(0, row - padding)
        minc = max(0, col - padding)
        maxr = min(image.shape[0], row + padding)
        maxc = min(image.shape[1], col + padding)
        
        # Save cropped region
        crop_img = image[minr:maxr, minc:maxc]
        output_file = Path(output_path) / f'{img_name}_{index}{self.config.jpeg_extension}'
        cv2.imwrite(str(output_file), crop_img)
        
        return region_data, str(output_file)
    
    def process_regions(self, regions: List, image: np.ndarray, img_name: str, 
                       output_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Process regions: filter, extract data, and save crops."""
        valid_regions = self.filter_regions(regions)
        if not valid_regions:
            return pd.DataFrame(columns=MEASUREMENT_COLUMNS), []
        
        # Extract data and save crops for all valid regions
        data_list = []
        output_paths = []
        for i, region in enumerate(valid_regions):
            region_data, output_file = self.extract_and_save_region(region, image, img_name, i, output_path)
            data_list.append(region_data)
            output_paths.append(output_file)
        
        return pd.DataFrame(data_list, columns=MEASUREMENT_COLUMNS), output_paths

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
        group_output_path = Path(output_path) / project / date / time / location
        group_output_path.mkdir(parents=True, exist_ok=True)
        
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
            output_file = group_output_path / f'objectMeasurements_{project}_{date}_{time}_{location}{self.config.csv_extension}'
            combined_df.to_csv(output_file, sep=self.config.csv_separator, index=False)
            print(f"[DETECTION]: Detection completed successfully! Total objects detected: {len(combined_df)}")
        else:
            print(f"[DETECTION]: Detection completed. No objects detected.")
        
        # Return sorted list of output image file paths
        return sorted(all_output_paths)

def main():
    """Command line interface for object detection."""    
    parser = argparse.ArgumentParser(description='Process images for object detection.')
    parser.add_argument('-i', '--image_folder', required=True, help='Path to the folder containing images to process')
    parser.add_argument('-o', '--output_path', default="./output/vignettes", help='Root output directory path where results will be saved')
    
    args = parser.parse_args()
    
    try:
        # Get image group from folder
        image_group = CommonCLI.get_image_group_from_folder(args.image_folder)
        
        # Validate output path
        output_path = CommonCLI.validate_output_path(args.output_path)
        
        # Process the image group
        detector = ObjectDetector()
        result_paths = detector.process_group(image_group, output_path)
        
        if result_paths:
            # Ensure the path shows as relative with leading './' if it's a relative path
            result_dir = os.path.dirname(result_paths[0])
            if not os.path.isabs(result_dir):
                result_dir = os.path.join('.', result_dir)
            print(f"[DETECTION]: Results saved to: {result_dir}")
        
    except Exception as e:
        print(f"[DETECTION]: Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())