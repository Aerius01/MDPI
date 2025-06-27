import cv2
import os
import pandas as pd
from imutils import paths
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from itertools import groupby
import numpy as np
from pathlib import Path

# === PROCESSING CONSTANTS ===
class ProcessingConfig:
    # Depth overlap correction
    DEPTH_MULTIPLIER = 100
    IMAGE_HEIGHT_CM = 4.3
    IMAGE_HEIGHT_PIXELS = 2048
    
    # Image thresholding
    THRESHOLD_VALUE = 190
    THRESHOLD_MAX = 255
    
    # Object filtering
    MIN_OBJECT_SIZE = 75
    MAX_OBJECT_SIZE = 5000
    
    # Region filtering
    MAX_ECCENTRICITY = 0.97
    MAX_MEAN_INTENSITY = 130
    MIN_MAJOR_AXIS_LENGTH = 25
    MAX_MIN_INTENSITY = 65
    
    # Object cropping
    SMALL_OBJECT_PADDING = 25
    MEDIUM_OBJECT_PADDING = 30
    LARGE_OBJECT_PADDING = 40
    SMALL_OBJECT_THRESHOLD = 40
    MEDIUM_OBJECT_THRESHOLD = 50
    
    # File operations
    CSV_SEPARATOR = ';'
    JPEG_EXTENSION = '.jpeg'
    CSV_EXTENSION = '.csv'
    BATCH_SIZE = 10

# Column definitions for consistent DataFrame creation
MEASUREMENT_COLUMNS = [
    'Filename', 'Area', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
    'Orientation', 'EquivDiameter', 'Solidity', 'Extent', 'MaxIntensity', 
    'MeanIntensity', 'MinIntensity', 'Perimeter'
]

def calculate_overlap_corrections(img_names):
    """Calculate pixel overlaps for depth correction."""
    depths = np.array([float(name.split('_')[0]) for name in img_names])
    image_bottom_depths = depths * ProcessingConfig.DEPTH_MULTIPLIER + ProcessingConfig.IMAGE_HEIGHT_CM
    image_top_depths = depths * ProcessingConfig.DEPTH_MULTIPLIER
    
    overlaps = np.zeros(len(depths))
    overlaps[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
    
    return np.round((overlaps / ProcessingConfig.IMAGE_HEIGHT_CM) * ProcessingConfig.IMAGE_HEIGHT_PIXELS).astype(int)

def apply_overlap_correction(img_bins, img_names):
    """Apply depth overlap correction to binary images."""
    pixel_overlaps = calculate_overlap_corrections(img_names)
    return [
        np.where(np.arange(img_bin.shape[0])[:, None] < pixel_overlap, 
                ProcessingConfig.THRESHOLD_MAX, img_bin)
        for img_bin, pixel_overlap in zip(img_bins, pixel_overlaps)
    ]

def create_empty_dataframe():
    """Create empty DataFrame with standard columns."""
    return pd.DataFrame(columns=MEASUREMENT_COLUMNS)

def filter_regions_by_criteria(regions):
    """Apply all filtering criteria to regions and return valid ones."""
    if not regions:
        return []
    
    # Create combined filter mask
    valid_mask = [
        (r.eccentricity < ProcessingConfig.MAX_ECCENTRICITY and
         r.mean_intensity < ProcessingConfig.MAX_MEAN_INTENSITY and
         r.major_axis_length > ProcessingConfig.MIN_MAJOR_AXIS_LENGTH and
         r.min_intensity < ProcessingConfig.MAX_MIN_INTENSITY)
        for r in regions
    ]
    
    return [r for r, valid in zip(regions, valid_mask) if valid]

def calculate_crop_padding(major_axis_length):
    """Calculate appropriate padding based on object size."""
    if major_axis_length < ProcessingConfig.SMALL_OBJECT_THRESHOLD:
        return ProcessingConfig.SMALL_OBJECT_PADDING
    elif major_axis_length < ProcessingConfig.MEDIUM_OBJECT_THRESHOLD:
        return ProcessingConfig.MEDIUM_OBJECT_PADDING
    else:
        return ProcessingConfig.LARGE_OBJECT_PADDING

def extract_and_save_region(region, image, img_name, index, output_path):
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
    padding = calculate_crop_padding(region.major_axis_length)
    
    minr = max(0, row - padding)
    minc = max(0, col - padding)
    maxr = min(image.shape[0], row + padding)
    maxc = min(image.shape[1], col + padding)
    
    # Save cropped region
    crop_img = image[minr:maxr, minc:maxc]
    output_file = Path(output_path) / f'{img_name}_{index}{ProcessingConfig.JPEG_EXTENSION}'
    cv2.imwrite(str(output_file), crop_img)
    
    return region_data

def process_regions(regions, image, img_name, output_path):
    """Process regions: filter, extract data, and save crops."""
    valid_regions = filter_regions_by_criteria(regions)
    if not valid_regions:
        return create_empty_dataframe()
    
    # Extract data and save crops for all valid regions
    data_list = [
        extract_and_save_region(region, image, img_name, i, output_path)
        for i, region in enumerate(valid_regions)
    ]
    
    return pd.DataFrame(data_list, columns=MEASUREMENT_COLUMNS)

def apply_size_filtering(label_imgs):
    """Apply size-based filtering to labeled images."""
    return [
        remove_small_objects(label_img > 0, min_size=ProcessingConfig.MIN_OBJECT_SIZE) & 
        ~remove_small_objects(label_img > 0, min_size=ProcessingConfig.MAX_OBJECT_SIZE)
        for label_img in label_imgs
    ]

def process_batch(batch_corrected_bins, batch_images, batch_names, group_output_path):
    """Process a batch of corrected binary images to detect and analyze objects."""
    # Perform object detection and size filtering
    batch_label_imgs = [label(img) for img in batch_corrected_bins]
    batch_filtered_imgs = apply_size_filtering(batch_label_imgs)
    
    # Extract regions and process each image
    batch_regions = [regionprops(label(img), image) 
                    for img, image in zip(batch_filtered_imgs, batch_images)]
    
    batch_data_dfs = []
    for region, image, img_name in zip(batch_regions, batch_images, batch_names):
        data_df = process_regions(region, image, img_name, group_output_path)
        if not data_df.empty:
            batch_data_dfs.append(data_df)
    
    # Explicit memory cleanup
    del batch_corrected_bins, batch_label_imgs, batch_filtered_imgs, batch_regions
    
    return batch_data_dfs

def parse_image_metadata(image_path):
    """Extract metadata from image filename."""
    filename = Path(image_path).stem
    _, project, date, time, location = filename.split('_')
    return project, date, time, location

def load_and_threshold_images(image_paths):
    """Load images and apply thresholding."""
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    img_bins = [cv2.threshold(img, ProcessingConfig.THRESHOLD_VALUE, 
                             ProcessingConfig.THRESHOLD_MAX, cv2.THRESH_BINARY_INV)[1] 
               for img in images]
    return images, img_bins

def process_image_group(image_group, output_path, group_index, total_groups):
    """Process a single group of images."""
    # Extract names and metadata
    img_names = [Path(path).stem for path in image_group]
    project, date, time, location = parse_image_metadata(image_group[0])
    
    print(f"[OBJECT DETECTION] Processing group {group_index+1}/{total_groups}: "
          f"{project}/{date}/{time}/{location} ({len(image_group)} images)")
    
    # Create output directory
    group_output_path = Path(output_path) / project / date / time / location
    group_output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and threshold images
    print(f"[OBJECT DETECTION] Loading and thresholding...")
    images, img_bins = load_and_threshold_images(image_group)
    
    # Step 2: Apply overlap correction
    print(f"[OBJECT DETECTION] Applying depth overlap correction...")
    corrected_img_bins = apply_overlap_correction(img_bins, img_names)
    
    # Step 3: Process in batches
    num_batches = int(np.ceil(len(image_group) / ProcessingConfig.BATCH_SIZE))
    print(f"[OBJECT DETECTION] Performing object detection in {num_batches} batches of {ProcessingConfig.BATCH_SIZE} images...")
    
    all_data_dfs = []
    for j in tqdm(range(0, len(image_group), ProcessingConfig.BATCH_SIZE)):
        batch_end = j + ProcessingConfig.BATCH_SIZE
        batch_corrected_bins = corrected_img_bins[j:batch_end]
        batch_images = images[j:batch_end]
        batch_names = img_names[j:batch_end]
        
        batch_data_dfs = process_batch(batch_corrected_bins, batch_images, 
                                     batch_names, str(group_output_path))
        all_data_dfs.extend(batch_data_dfs)
    
    # Cleanup group-level memory
    del images, img_bins, corrected_img_bins
    
    # Save results
    if all_data_dfs:
        combined_df = pd.concat(all_data_dfs, ignore_index=True)
        output_file = group_output_path / f'objectMeasurements_{project}_{date}_{time}_{location}{ProcessingConfig.CSV_EXTENSION}'
        combined_df.to_csv(output_file, sep=ProcessingConfig.CSV_SEPARATOR, index=False)
        print(f"[OBJECT DETECTION] Group {group_index+1} completed. Total objects detected: {len(combined_df)}")
    else:
        print(f"[OBJECT DETECTION] Group {group_index+1} completed. No objects detected.")

def detect_objects(input_path, output_path):
    """Main function to detect objects in all image groups."""
    print(f"[OBJECT DETECTION] Starting object detection...")
    
    # Group and sort images by directory
    image_paths = list(paths.list_images(input_path))
    image_groups = [
        sorted(list(group)) 
        for key, group in groupby(sorted(image_paths, key=os.path.dirname), os.path.dirname)
    ]
    
    print(f"[OBJECT DETECTION] Processing {len(image_groups)} image groups...")
    
    # Process each image group
    for i, image_group in enumerate(image_groups):
        process_image_group(image_group, output_path, i, len(image_groups))
    
    print(f"[OBJECT DETECTION] Object detection completed successfully!")