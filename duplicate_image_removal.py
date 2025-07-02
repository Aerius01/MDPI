import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from tools.hash.image_hashing import ImageHash
from constants import get_image_sort_key
from typing import List, Dict
from dataclasses import dataclass
from cli_utils import CommonCLI

@dataclass
class DuplicateConfig:
    """Configuration for duplicate detection."""
    remove: bool = False
    display_size: tuple = (500, 500)
    show_montages: bool = True

class DuplicateDetector:
    """Handles duplicate image detection and removal."""
    
    def __init__(self, config: DuplicateConfig):
        self.config = config
        self.image_hasher = ImageHash()
    
    def _compute_hashes(self, image_paths: List[str]) -> Dict[str, List[str]]:
        """Compute hashes for all images and group by hash."""
        hashes = {}
        print("[DUPLICATES]: Computing image hashes...")
        
        for image_path in tqdm(image_paths, desc='[DUPLICATES]'):
            image = cv2.imread(image_path)
            hash_value = self.image_hasher.dhash(image)
            hashes.setdefault(hash_value, []).append(image_path)
        
        return hashes
    
    def _create_montage(self, image_paths: List[str]) -> np.ndarray:
        """Create horizontal montage from image paths."""
        montage = None
        for path in image_paths:
            image = cv2.imread(path)
            image = cv2.resize(image, self.config.display_size)
            montage = image if montage is None else np.hstack([montage, image])
        return montage
    
    def _handle_duplicates(self, hash_value: str, duplicate_paths: List[str]) -> int:
        """Handle duplicate images based on configuration."""
        if not self.config.remove:
            if self.config.show_montages:
                montage = self._create_montage(duplicate_paths)
                print(f"[INFO] hash: {hash_value}")
                cv2.imshow("Montage", montage)
                cv2.waitKey(0)
            return 0
        else:
            # Remove all but the first image
            for path in duplicate_paths[1:]:
                os.remove(path)
            return len(duplicate_paths) - 1
    
    def process_group(self, image_paths: List[str]) -> int:
        """Main method to detect and process duplicates."""
        hashes = self._compute_hashes(image_paths)
        
        print('[DUPLICATES]: Detecting duplicate images...')
        total_removed = 0
        
        duplicate_hashes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
        
        for hash_value, duplicate_paths in duplicate_hashes.items():
            total_removed += self._handle_duplicates(hash_value, duplicate_paths)
        
        print(f'[DUPLICATES]: {total_removed} duplicate images removed')
        return total_removed

def main():
    """Command line interface for duplicate detection."""
    parser = argparse.ArgumentParser(description='Detect and remove duplicate images from a folder.')
    parser.add_argument('-i', '--image_folder', required=True, help='Path to the folder containing images to process')
    parser.add_argument('-r', '--remove', action='store_true', help='Remove duplicate images (default: only detect and show)')
    
    args = parser.parse_args()
    
    try:
        # Validate input folder exists
        if not os.path.exists(args.image_folder):
            raise FileNotFoundError(f"Image folder does not exist: {args.image_folder}")
        
        if not os.path.isdir(args.image_folder):
            raise ValueError(f"Path is not a directory: {args.image_folder}")
        
        # Get image group using the module's own sorting method
        image_group = CommonCLI.get_image_group_from_folder(args.image_folder, get_image_sort_key)
        
        if not image_group:
            raise ValueError(f"No images found in folder: {args.image_folder}")
        
        # Create configuration
        config = DuplicateConfig(
            remove=args.remove,
            show_montages=True
        )
        
        # Process the image group
        detector = DuplicateDetector(config)
        total_processed = detector.process_group(image_group)
        
        # Print results
        if args.remove:
            if total_processed > 0:
                print(f"[DUPLICATES]: Duplicates removed from: {args.image_folder}")
        else:
            if total_processed > 0:
                print(f"[DUPLICATES]: Rerun with --remove flag to delete duplicates")
            else:
                print(f"[DUPLICATES]: No duplicates found in: {args.image_folder}")
        
    except Exception as e:
        print(f"[DUPLICATES]: Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())