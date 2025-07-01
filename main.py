from duplicate_image_removal import remove_duplicate_images
from depth_profiler import profile_depths
from flatfielding_profiles import flatfielding_profiles
from dateutil import relativedelta
from object_detection import detect_objects
from object_classification import classify_objects
import os
import argparse

# Physical camera capture rate is 2.4 Hz due to data transfer from network connection to computer
# if this is not expressed as microseconds, the comparison will not be sensitive enough to capture the time difference
TIMESTEP = relativedelta.relativedelta(microseconds=1/2.4*1000000)

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

    # Self-explanatory
    remove_duplicate_images(dataset_path, remove=False)

    # Add the depth value to the image file names
    profile_depths(dataset_path, depth_profiles_path, TIMESTEP)

    # Flatfield the images
    flatfielding_profiles(depth_profiles_path, flatfield_profiles_path)

    # Detect objects in the images
    detect_objects(flatfield_profiles_path, object_detection_path)

    # Classify objects in the images
    classify_objects(object_detection_path, model_path, output_path=object_classification_path)

if __name__ == "__main__":
    main()