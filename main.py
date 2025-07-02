from duplicate_image_removal import DuplicateDetector, DuplicateConfig
from depth_profiler import DepthProfiler
from flatfielding_profiles import FlatfieldProcessor
from object_detection import ObjectDetector
from object_classification import classify_objects
import os
import argparse
from imutils import paths
from itertools import groupby
from constants import get_image_sort_key
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PipelineConfig:
    """Centralized configuration for the processing pipeline."""
    root_output_path: str
    dataset_path: str
    model_path: str
    
    @property
    def output_paths(self) -> dict:
        """Generate all output paths from root."""
        return {
            'depth_profiles': os.path.join(self.root_output_path, "depth_profiles"),
            'flatfield_profiles': os.path.join(self.root_output_path, "flatfield_profiles"),
            'object_detection': os.path.join(self.root_output_path, "vignettes"),
            'object_classification': os.path.join(self.root_output_path, "classification")
        }

class ImagePipeline:
    """Encapsulates the complete image processing pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.paths = config.output_paths

        # Create output directory if it doesn't exist
        os.makedirs(self.config.root_output_path, exist_ok=True)

        # Initialize processors
        self.duplicate_detector = DuplicateDetector(DuplicateConfig(remove=False, show_montages=False))
        self.depth_profiler = DepthProfiler()
        self.flatfield_processor = FlatfieldProcessor()
        self.object_detector = ObjectDetector()
    
    def _get_sort_key(self, path: str) -> int:
        """Extract sort key from image path."""
        return get_image_sort_key(path)
    
    def _get_or_fallback_images(self, images: Optional[List[str]], fallback_path: str) -> List[str]:
        """Get images list or fallback to directory listing with error handling."""
        if images:
            return images
        try:
            return sorted(list(paths.list_images(fallback_path)))
        except FileNotFoundError:
            raise FileNotFoundError(f"No images found in {fallback_path}")
    
    def process_group(self, group: List[str], group_index: int, total_groups: int):
        """Process a single image group through the complete pipeline."""
        print(f"\n[MAIN]: Processing image group {group_index+1}/{total_groups}: {os.path.dirname(group[0])}")
        
        # Pipeline stages with fallback handling
        self.duplicate_detector.process_group(group)
        
        profiled_images = self.depth_profiler.process_group(group, self.paths['depth_profiles'])
        profiled_images = self._get_or_fallback_images(profiled_images, self.paths['depth_profiles'])
        
        flatfielded_images = self.flatfield_processor.process_group(profiled_images, self.paths['flatfield_profiles'])
        flatfielded_images = self._get_or_fallback_images(flatfielded_images, self.paths['flatfield_profiles'])
        
        vignette_images = self.object_detector.process_group(flatfielded_images, self.paths['object_detection'])
        vignette_images = self._get_or_fallback_images(vignette_images, self.paths['object_detection'])
        
        classify_objects(vignette_images, self.paths['object_classification'], self.config.model_path)
    
    def run(self):
        """Execute the complete pipeline on all image groups."""
        image_paths = list(paths.list_images(self.config.dataset_path))
        
        # Group and sort images efficiently
        image_groups = [
            sorted(list(group), key=self._get_sort_key) 
            for key, group in groupby(sorted(image_paths, key=os.path.dirname), os.path.dirname)
        ]
        
        # Process each group
        for i, group in enumerate(image_groups):
            self.process_group(group, i, len(image_groups))

def main():
    parser = argparse.ArgumentParser(description='Process images for depth profiling, flatfielding, object detection, and classification.')
    parser.add_argument('-o', '--root_output_path', default="./output", help='Root output directory path (default: ./output)')
    parser.add_argument('-d', '--dataset_path', default="./profiles", help='Dataset directory path (default: ./profiles)')
    parser.add_argument('-m', '--model_path', default="./model", help='Model directory path (default: ./model)')
    
    args = parser.parse_args()
    config = PipelineConfig(args.root_output_path, args.dataset_path, args.model_path)
    
    pipeline = ImagePipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()