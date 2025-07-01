import numpy as np
import cv2
import os
from tqdm import tqdm
from tools.hash.image_hashing import ImageHash
from typing import List, Dict
from dataclasses import dataclass

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
    
    def detect_and_process(self, image_paths: List[str]) -> int:
        """Main method to detect and process duplicates."""
        hashes = self._compute_hashes(image_paths)
        
        print('[DUPLICATES]: Detecting duplicate images...')
        total_removed = 0
        
        duplicate_hashes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
        
        for hash_value, duplicate_paths in duplicate_hashes.items():
            total_removed += self._handle_duplicates(hash_value, duplicate_paths)
        
        print(f'[DUPLICATES]: {total_removed} duplicate images removed')
        return total_removed