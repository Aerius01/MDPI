import cv2
import os
from tqdm import tqdm
from tools.hash.image_hashing import ImageHash
from typing import List, Dict

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

def _handle_duplicates(duplicate_paths: List[str]) -> List[str]:
    """Handle duplicate images based on configuration."""
    # Remove all but the first image
    paths_to_remove = duplicate_paths[1:]
    for path in paths_to_remove:
        os.remove(path)
    return paths_to_remove
    
# This is the heart of the duplicate detection module. It operates upon a group of image paths, 
# directly modifying the directory where those images are stored.
def deduplicate_images(image_paths: List[str]) -> List[str]:
    """
    Main method to detect and process duplicates in a list of image paths.
    
    Args:
        image_paths: A list of image paths to deduplicate.
        
    Returns:
        A list of paths for the duplicate images that were removed.
    """
    hashes = _compute_hashes(image_paths)
    
    print('[DUPLICATES]: Detecting duplicate images...')
    
    removed_paths = []
    duplicate_hashes = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    for duplicate_paths in duplicate_hashes.values():
        removed_in_group = _handle_duplicates(
            duplicate_paths
        )
        removed_paths.extend(removed_in_group)
        
    total_removed = len(removed_paths)
    print(f'[DUPLICATES]: {total_removed} duplicate images removed')

    return removed_paths