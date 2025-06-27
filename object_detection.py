import cv2
import os
import pandas as pd
from imutils import paths
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from itertools import groupby
import numpy as np

# Constants for depth overlap correction
DEPTH_MULTIPLIER = 100  # Convert depth to cm
IMAGE_HEIGHT_CM = 4.3   # Height of each image in cm
IMAGE_HEIGHT_PIXELS = 2048  # Height of each image in pixels

# Constants for image thresholding
THRESHOLD_VALUE = 190
THRESHOLD_MAX = 255

# Constants for object filtering
MIN_OBJECT_SIZE = 75
MAX_OBJECT_SIZE = 5000

# Constants for region filtering
MAX_ECCENTRICITY = 0.97
MAX_MEAN_INTENSITY = 130
MIN_MAJOR_AXIS_LENGTH = 25
MAX_MIN_INTENSITY = 65

# Constants for object cropping
SMALL_OBJECT_PADDING = 25
MEDIUM_OBJECT_PADDING = 30
LARGE_OBJECT_PADDING = 40
SMALL_OBJECT_THRESHOLD = 40
MEDIUM_OBJECT_THRESHOLD = 50

# Constants for file operations
CSV_SEPARATOR = ';'
JPEG_EXTENSION = '.jpeg'
CSV_EXTENSION = '.csv'

# Constants for batching
BATCH_SIZE = 10

def overlap_correction(img_bins, img_names):
    """
    Apply depth overlap correction to binary images.
    
    Args:
        img_bins: List of binary images
        img_names: List of image names containing depth information
    
    Returns:
        List of corrected binary images
    """
    # Extract depths from filenames
    depths = np.array([float(img_name.split('_')[0]) for img_name in img_names])
    image_bottom_depths = depths * DEPTH_MULTIPLIER + IMAGE_HEIGHT_CM
    image_top_depths = depths * DEPTH_MULTIPLIER
    
    # Calculate overlaps: previous image bottom - current image top
    # For first image, overlap is 0
    overlaps = np.zeros(len(depths))
    overlaps[1:] = np.maximum(0, image_bottom_depths[:-1] - image_top_depths[1:])
    
    # Convert depth overlaps to pixel overlaps
    pixel_overlaps = np.round((overlaps / IMAGE_HEIGHT_CM) * IMAGE_HEIGHT_PIXELS).astype(int)
    
    # Apply overlap corrections
    corrected_img_bins = [
        np.where(np.arange(img_bin.shape[0])[:, None] < pixel_overlap, THRESHOLD_MAX, img_bin)
        for img_bin, pixel_overlap in zip(img_bins, pixel_overlaps)
    ]
    
    return corrected_img_bins

def process_regions(regions, image, img_name, outputPath):
    if not regions:
        return pd.DataFrame(columns=['Filename', 'Area', 'MajorAxisLength',
                                    'MinorAxisLength', 'Eccentricity',
                                    'Orientation', 'EquivDiameter', 'Solidity',
                                    'Extent', 'MaxIntensity', 'MeanIntensity',
                                    'MinIntensity', 'Perimeter'])
    
    # Create individual filter masks and then combine them
    eccentricity_mask = [r.eccentricity < MAX_ECCENTRICITY for r in regions]
    mean_intensity_mask = [r.mean_intensity < MAX_MEAN_INTENSITY for r in regions]
    major_axis_mask = [r.major_axis_length > MIN_MAJOR_AXIS_LENGTH for r in regions]
    min_intensity_mask = [r.min_intensity < MAX_MIN_INTENSITY for r in regions]
    valid_mask = [eccentricity_valid and mean_intensity_valid and major_axis_valid and min_intensity_valid 
                  for eccentricity_valid, mean_intensity_valid, major_axis_valid, min_intensity_valid in 
                  zip(eccentricity_mask, mean_intensity_mask, major_axis_mask, min_intensity_mask)]
    
    # Filter out invalid regions
    valid_regions = [r for r, valid in zip(regions, valid_mask) if valid]
    if not valid_regions:
        return pd.DataFrame(columns=['Filename', 'Area', 'MajorAxisLength',
                                    'MinorAxisLength', 'Eccentricity',
                                    'Orientation', 'EquivDiameter', 'Solidity',
                                    'Extent', 'MaxIntensity', 'MeanIntensity',
                                    'MinIntensity', 'Perimeter'])
    
    # Crop regions and save them
    data_list = []
    for i, region in enumerate(valid_regions):
        data_list.append([
            f"{img_name}_{i}",
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
        ])

        row, col = region.centroid
        row = int(row)
        col = int(col)

        # Set the padding size
        if region.major_axis_length < SMALL_OBJECT_THRESHOLD:
            padding = SMALL_OBJECT_PADDING
        elif region.major_axis_length < MEDIUM_OBJECT_THRESHOLD:
            padding = MEDIUM_OBJECT_PADDING
        else:
            padding = LARGE_OBJECT_PADDING

        # add padding
        minr = 0 if row - padding < 0 else row - padding
        minc = 0 if col - padding < 0 else col - padding
        maxr = image.shape[0] if row + padding > image.shape[0] else row + padding
        maxc = image.shape[1] if col + padding > image.shape[1] else col + padding

        # crop & save the region of interest
        crop_img = image[minr:maxr, minc:maxc]
        cv2.imwrite(os.path.sep.join([outputPath, f'{img_name}_{i}{JPEG_EXTENSION}']), crop_img)
    
    # Create DataFrame from collected data
    data_df = pd.DataFrame(data_list, columns=['Filename', 'Area', 'MajorAxisLength',
                                               'MinorAxisLength', 'Eccentricity',
                                               'Orientation', 'EquivDiameter', 'Solidity',
                                               'Extent', 'MaxIntensity', 'MeanIntensity',
                                               'MinIntensity', 'Perimeter'])
    
    return data_df

def process_batch(batch_corrected_bins, batch_images, batch_names, group_output_path):
    """
    Process a batch of corrected binary images to detect and analyze objects.
    
    Args:
        batch_corrected_bins: List of corrected binary images
        batch_images: List of original grayscale images
        batch_names: List of image names
        group_output_path: Output path for the group
    
    Returns:
        List of DataFrames containing object measurements
    """
    # measure objects after thresholding and remove smaller-than and larger-than objects
    batch_label_imgs = [label(img) for img in batch_corrected_bins]
    
    # Convert labeled images to boolean masks and apply size filtering
    batch_filtered_imgs = [
        remove_small_objects(label_img > 0, min_size=MIN_OBJECT_SIZE) & 
        ~remove_small_objects(label_img > 0, min_size=MAX_OBJECT_SIZE)
        for label_img in batch_label_imgs
    ]
    
    # regions is a list of lists of region objects. Each sub-list corresponds to the regions in a single image
    batch_regions = [regionprops(label(img), image) for img, image in zip(batch_filtered_imgs, batch_images)]
    
    # Process each image in the batch
    batch_data_dfs = []
    for region, image, img_name in zip(batch_regions, batch_images, batch_names):
        data_df = process_regions(region, image, img_name, group_output_path)
        if not data_df.empty:
            batch_data_dfs.append(data_df)
    
    # Explicitly free memory for this batch
    del batch_corrected_bins, batch_label_imgs, batch_filtered_imgs, batch_regions
    
    return batch_data_dfs

def detect_objects(inputPath, outputPath):
    print(f"[OBJECT DETECTION] Starting object detection...")

    # Get all image paths in all subdirectories. First group them by directory, then sort each group
    imagePaths = list(paths.list_images(inputPath))
    imageGroups = [list(group) for key, group in groupby(sorted(imagePaths, key=os.path.dirname), os.path.dirname)]
    imageGroups = [sorted(group) for group in imageGroups]
    
    print(f"[OBJECT DETECTION] Processing {len(imageGroups)} image groups...")

    for i, imageGroup in enumerate(imageGroups):
        # Image names without their extensions
        img_names = [os.path.splitext(os.path.basename(path))[0] for path in imageGroup]
        
        # Extract metadata, using the first image's metadata since all images in the group have the same metadata
        filename = os.path.basename(imageGroup[0])
        _, project, date, time, location = os.path.splitext(filename)[0].split('_')

        print(f"[OBJECT DETECTION] Processing group {i+1}/{len(imageGroups)}: {project}/{date}/{time}/{location} ({len(imageGroup)} images)")

        # Create output path
        group_output_path = os.path.sep.join([outputPath, project, date, time, location])
        os.makedirs(group_output_path, exist_ok=True)

        # Step 1: Load all images and apply thresholding for the entire group
        print(f"[OBJECT DETECTION] Loading and thresholding...")
        images = [cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE) for imagePath in imageGroup]
        img_bins = [cv2.threshold(img, THRESHOLD_VALUE, THRESHOLD_MAX, cv2.THRESH_BINARY_INV)[1] for img in images]

        # Step 2: Apply depth overlap correction to the entire group
        print(f"[OBJECT DETECTION] Applying depth overlap correction...")
        corrected_img_bins = overlap_correction(img_bins, img_names)

        # Step 3: Process corrected images in batches to avoid memory issues
        print(f"[OBJECT DETECTION] Performing object detection in {int(np.ceil(len(imageGroup)/BATCH_SIZE))} batches of {BATCH_SIZE} images...")
        all_data_dfs = []
        
        for j in tqdm(range(0, len(imageGroup), BATCH_SIZE)):
            # Get batch of corrected images and original images
            batch_corrected_bins = corrected_img_bins[j:j + BATCH_SIZE]
            batch_images = images[j:j + BATCH_SIZE]
            batch_names = img_names[j:j + BATCH_SIZE]

            # Process the batch
            batch_data_dfs = process_batch(batch_corrected_bins, batch_images, batch_names, group_output_path)
            
            # Collect DataFrames from this batch
            all_data_dfs.extend(batch_data_dfs)
        
        # Free memory for the entire group
        del images, img_bins, corrected_img_bins
        
        # Combine all DataFrames from all batches for this image group
        if all_data_dfs:
            combined_df = pd.concat(all_data_dfs, ignore_index=True)
            # Save combined measurements for this group
            combined_df.to_csv(os.path.sep.join([group_output_path, f'objectMeasurements_{project}_{date}_{time}_{location}{CSV_EXTENSION}']),
                              sep=CSV_SEPARATOR, index=False)
            print(f"[OBJECT DETECTION] Group {i+1} completed. Total objects detected: {len(combined_df)}")
        else:
            print(f"[OBJECT DETECTION] Group {i+1} completed. No objects detected.")
    
    print(f"[OBJECT DETECTION] Object detection completed successfully!")