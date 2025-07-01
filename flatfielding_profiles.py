import cv2
import os
import numpy as np
from tqdm import tqdm
from constants import BATCH_SIZE, NORMALIZATION_FACTOR

def flatfielding_profiles(image_group, output_path):
    print(f"[FLATFIELDING]: Starting flatfielding...")
    
    # Get image names without extension for saving later
    img_names = [os.path.splitext(os.path.basename(path))[0] for path in image_group]
    
    # Extract metadata from filename: depth_project_date_time_location.tiff
    # Use the first image's metadata since all images in the group should have the same metadata
    filename = os.path.basename(image_group[0])
    _, project, date, time, location = os.path.splitext(filename)[0].split('_')

    print(f"[FLATFIELDING]: Processing group: {project}/{date}/{time}/{location} ({len(image_group)} images)")

    # create output path
    outputPath = os.path.sep.join([output_path, project, date, time, location])
    os.makedirs(outputPath, exist_ok=True)

    # Calculate the average image to perform flatfielding
    print(f"[FLATFIELDING]: Calculating average image...")
    images = np.array([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in image_group])
    ff = np.average(images, axis=0).astype('uint8')

    # List to store all output file paths
    output_file_paths = []

    # Flatfield images in batches to avoid memory issues with divide and clip operations
    print(f"[FLATFIELDING]: Flatfielding images in {int(np.ceil(len(image_group)/BATCH_SIZE))} batches...")
    for j in tqdm(range(0, len(image_group), BATCH_SIZE), desc='[FLATFIELDING]'):
        batch_images = images[j:j + BATCH_SIZE]
        batch_names = img_names[j:j + BATCH_SIZE]
        
        # Perform flatfielding on batch (divide and clip operations)
        flatfielded_batch = np.divide(batch_images, ff) * NORMALIZATION_FACTOR
        flatfielded_batch = np.clip(flatfielded_batch, 0, 255).astype('uint8')
        
        # Save batch of flatfielded images
        for img, name in zip(flatfielded_batch, batch_names):
            output_file_path = os.path.sep.join([outputPath, f'{name}.jpeg'])
            cv2.imwrite(output_file_path, img)
            output_file_paths.append(output_file_path)
        
        # Explicitly free memory
        del flatfielded_batch

    print(f"[FLATFIELDING]: Flatfielding completed successfully!")
    
    # Return all output file paths sorted by filename
    return sorted(output_file_paths)
