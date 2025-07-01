from duplicate_image_removal import remove_duplicate_images
from depth_profiler import profile_depths
from flatfielding_profiles import flatfielding_profiles
from object_detection import detect_objects
from object_classification import classify_objects
import os
import argparse
from imutils import paths
from itertools import groupby
from constants import BASE_FILENAME_PATTERN

def _get_sort_key(path):
    filename = os.path.basename(path)
    match = BASE_FILENAME_PATTERN.search(filename)
    return int(match.group(2)) if match else 0

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Process images for depth profiling, flatfielding, object detection, and classification.')
    parser.add_argument('-o', '--root_output_path', default=".", 
                       help='Root output directory path (default: ./output)')
    parser.add_argument('-d', '--dataset_path', default="./profiles", 
                       help='Dataset directory path (default: ./profiles)')
    parser.add_argument('-m', '--model_path', default="./model", 
                       help='Model directory path (default: ./model)')
    
    args = parser.parse_args()
    
    # Use command line arguments or defaults
    root_output_path = args.root_output_path
    dataset_path = args.dataset_path
    model_path = args.model_path

    # Automatically generated
    depth_profiles_path = os.path.join(root_output_path, "depth_profiles")
    flatfield_profiles_path = os.path.join(root_output_path, "flatfield_profiles")
    object_detection_path = os.path.join(root_output_path, "vignettes")
    object_classification_path = os.path.join(root_output_path, "classification")

    # listing all images
    image_paths = list(paths.list_images(dataset_path))
    
    # First group by directory, then sort each group
    image_groups = [list(group) for key, group in groupby(sorted(image_paths, key=os.path.dirname), os.path.dirname)]
    image_groups = [sorted(group, key=_get_sort_key) for group in image_groups]

    for i, group in enumerate(image_groups):
        print(f"\n[MAIN]: Processing image group {i+1}/{len(image_groups)}: {os.path.dirname(group[0])}")
        # Self-explanatory
        remove_duplicate_images(group, remove=False)

        # Add the depth value to the image file names
        profiled_images = profile_depths(group, depth_profiles_path)

        # Flatfield the images
        if not profiled_images:
            try:
                profiled_images = sorted(list(paths.list_images(depth_profiles_path)))
            except FileNotFoundError:
                raise FileNotFoundError(f"No depth profiles found in {depth_profiles_path}")
        flatfielded_images = flatfielding_profiles(profiled_images, flatfield_profiles_path)

        # Detect objects in the images
        if not flatfielded_images:
            try:
                flatfielded_images = sorted(list(paths.list_images(flatfield_profiles_path)))
            except FileNotFoundError:
                raise FileNotFoundError(f"No flatfielded images found in {flatfield_profiles_path}")
        vignette_images = detect_objects(flatfielded_images, object_detection_path)

        # Classify objects in the images
        if not vignette_images:
            try:
                vignette_images = sorted(list(paths.list_images(object_detection_path)))
            except FileNotFoundError:
                raise FileNotFoundError(f"No vignette images found in {object_detection_path}")
        classify_objects(vignette_images, object_classification_path, model_path)

if __name__ == "__main__":
    main()