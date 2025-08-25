import numpy as np
import cv2
import os
from tqdm import tqdm
from tools.hash.image_hashing import ImageHash
from typing import List, Dict, Tuple
from modules.duplicate_detection.__main__ import DeduplicationData

def _compute_hashes(image_paths: List[str]) -> Dict[str, List[str]]:
    """Compute hashes for all images and group by hash."""
    image_hasher = ImageHash()
    hashes = {}
    print("[DUPLICATES]: Computing image hashes...")
    
    for image_path in tqdm(image_paths, desc='[DUPLICATES]'):
        image = cv2.imread(image_path)
        hash_value = image_hasher.dhash(image)
        hashes.setdefault(hash_value, []).append(image_path)
    
    return hashes

def _create_montage(image_paths: List[str], display_size: Tuple[int, int]) -> np.ndarray:
    """Create horizontal montage from image paths."""
    montage = None
    for path in image_paths:
        image = cv2.imread(path)
        image = cv2.resize(image, display_size)
        montage = image if montage is None else np.hstack([montage, image])
    return montage

def _handle_duplicates(hash_value: str, duplicate_paths: List[str], display_size: Tuple[int, int], remove: bool, show_montages: bool) -> List[str]:
    """Handle duplicate images based on configuration."""
    if not remove:
        if show_montages:
            montage = _create_montage(duplicate_paths, display_size)
            print(f"[INFO] hash: {hash_value}")
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)
        return []
    else:
        # Remove all but the first image
        paths_to_remove = duplicate_paths[1:]
        for path in paths_to_remove:
            os.remove(path)
        return paths_to_remove
    
# This is the heart of the duplicate detection module. It operates upon a group of image paths, 
# directly modifying the directory where those images are stored.
def deduplicate_images(deduplication_data: DeduplicationData) -> List[str]:
    """
    Main method to detect and process duplicates in a list of image paths.
    
    Args:
        deduplication_data: An object containing all the required data for deduplication.
        
    Returns:
        A list of paths for the duplicate images that were removed.
    """
    hashes = _compute_hashes(deduplication_data.image_paths)
    
    print('[DUPLICATES]: Detecting duplicate images...')
    
    removed_paths = []
    duplicate_hashes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    for hash_value, duplicate_paths in duplicate_hashes.items():
        removed_in_group = _handle_duplicates(
            hash_value, 
            duplicate_paths, 
            deduplication_data.display_size, 
            deduplication_data.remove, 
            deduplication_data.show_montages
        )
        removed_paths.extend(removed_in_group)
        
    total_removed = len(removed_paths)
    print(f'[DUPLICATES]: {total_removed} duplicate images removed')

    return removed_paths