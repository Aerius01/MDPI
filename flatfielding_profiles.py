import cv2
import os
import numpy as np
from imutils import paths
from itertools import groupby
from tqdm import tqdm

NORMALIZATION_FACTOR = 235
BATCH_SIZE = 10  # Process images in batches to reduce memory usage

def flatfielding_profiles(input_path, output_path):
    print(f"[FLATFIELDING] Starting flatfielding...")
    
    # The images now have the depth value as the first part of their filename, and so normal sorting will work
    imagePaths = list(paths.list_images(input_path))
    imagePathsSorted = sorted(imagePaths)

    # Group images by their directory paths
    imageGroups = [list(group) for key, group in groupby(imagePathsSorted, os.path.dirname)]
    print(f"[FLATFIELDING] Processing {len(imageGroups)} image groups...")
    
    for i, imageGroup in enumerate(imageGroups):
        # Get image names without extension for saving later
        img_names = [os.path.splitext(os.path.basename(path))[0] for path in imageGroup]
        
        # Extract metadata from filename: depth_project_date_time_location.tiff
        # Use the first image's metadata since all images in the group should have the same metadata
        filename = os.path.basename(imageGroup[0])
        _, project, date, time, location = os.path.splitext(filename)[0].split('_')

        print(f"[FLATFIELDING] Processing group {i+1}/{len(imageGroups)}: {project}/{date}/{time}/{location} ({len(imageGroup)} images)")

        # create output path
        outputPath = os.path.sep.join([output_path, project, date, time, location])
        os.makedirs(outputPath, exist_ok=True)

        # Calculate the average image to perform flatfielding
        print(f"[FLATFIELDING] Calculating average image...")
        images = np.array([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in imageGroup])
        ff = np.average(images, axis=0).astype('uint8')

        # Flatfield images in batches to avoid memory issues with divide and clip operations
        print(f"[FLATFIELDING] Flatfielding images in {int(np.ceil(len(imageGroup)/BATCH_SIZE))} batches...")
        for j in tqdm(range(0, len(imageGroup), BATCH_SIZE)):
            batch_images = images[j:j + BATCH_SIZE]
            batch_names = img_names[j:j + BATCH_SIZE]
            
            # Perform flatfielding on batch (divide and clip operations)
            flatfielded_batch = np.divide(batch_images, ff) * NORMALIZATION_FACTOR
            flatfielded_batch = np.clip(flatfielded_batch, 0, 255).astype('uint8')
            
            # Save batch of flatfielded images
            for img, name in zip(flatfielded_batch, batch_names):
                cv2.imwrite(os.path.sep.join([outputPath, f'{name}.jpeg']), img)
            
            # Explicitly free memory
            del flatfielded_batch

    print(f"[FLATFIELDING] Flatfielding completed successfully!")
