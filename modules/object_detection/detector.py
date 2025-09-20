import cv2
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
from tqdm import tqdm

from .detection_data import DetectionData


@dataclass
class ProcessedRegion:
    """Holds data for a single processed region."""
    region_id: int
    region_extents: Tuple[int, int, int, int]
    region_data: dict

@dataclass
class MappedImageRegions:
    """Holds the source image and its detected regions."""
    source_image_path: str
    source_image: np.ndarray
    processed_regions: List[ProcessedRegion]

def detect_objects(
    data: DetectionData,
):
    """
    Run the object detection process over flatfielded images and save outputs.
    """
    image_paths = data.flatfield_img_paths
    print(f"[DETECTION]: Found {len(image_paths)} flatfielded images")

    num_batches = int(np.ceil(len(image_paths) / data.batch_size))
    print(f"[DETECTION]: Performing object detection in {num_batches} batches...")

    all_region_data = []
    output_count = 0

    for i in tqdm(range(0, len(image_paths), data.batch_size), desc='[DETECTION]'):
        batch_end = i + data.batch_size
        batch_image_paths = image_paths[i:batch_end]

        batch_images, batch_binary_images = load_and_threshold_images(
            batch_image_paths, 
            data.threshold_value, 
            data.threshold_max
        )
        mapped_regions_batch = _detect_objects_in_batch(batch_images, batch_binary_images, batch_image_paths, data)

        for mapped_region in mapped_regions_batch:
            process_vignette_generator = process_vignette(mapped_region, data.output_path)
            for region_data, vignette_img, vignette_path in process_vignette_generator:
                all_region_data.append(region_data)
                cv2.imwrite(vignette_path, vignette_img)
                output_count += 1

    combined_df = create_dataframe(all_region_data, data.depth_profiles_df)
    save_dataframe(combined_df, output_count, data.output_path, data.csv_extension, data.csv_separator)

    print(f"[DETECTION]: Processing completed successfully!")
    print(f"[DETECTION]: {output_count} vignettes saved to {data.output_path}")

def load_and_threshold_images(
    image_paths: List[str], 
    threshold_value: int, 
    threshold_max: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load images and apply thresholding."""
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    
    img_bins = [cv2.threshold(img, threshold_value, threshold_max, cv2.THRESH_BINARY_INV)[1] 
                for img in images]
    
    return images, img_bins

def _apply_size_filtering(
    label_imgs: List[np.ndarray], 
    min_object_size: int, 
    max_object_size: int
) -> List[np.ndarray]:
    """Apply size-based filtering to labeled images."""
    return [
        remove_small_objects(label_img > 0, min_size=min_object_size) & 
        ~remove_small_objects(label_img > 0, min_size=max_object_size)
        for label_img in label_imgs
    ]

def _detect_objects_in_batch(images: List[np.ndarray], binary_images: List[np.ndarray], image_paths: List[str], data: DetectionData) -> List[MappedImageRegions]:
    """Process a batch of images to detect and analyze objects."""
    labeled_images = [label(img) for img in binary_images]
    filtered_images = _apply_size_filtering(
        labeled_images, 
        data.min_object_size, 
        data.max_object_size
    )
    
    regions = [regionprops(label(img), image) for img, image in zip(filtered_images, images)]

    processed_regions_by_image = [process_regions(regions[i], images[i].shape, data) for i in range(len(images))]
    
    mapped_image_regions = [
        MappedImageRegions(
            source_image_path=image_paths[i],
            source_image=images[i],
            processed_regions=processed_regions_by_image[i]
        ) 
        for i in range(len(image_paths))
    ]
    
    return mapped_image_regions

def filter_regions(
    regions: List, 
    max_eccentricity: float,
    max_mean_intensity: int,
    min_major_axis_length: int,
    max_min_intensity: int
) -> List:
    """Apply all filtering criteria to regions and return valid ones."""
    if not regions:
        return []
    
    return [
        r for r in regions
        if (r.eccentricity < max_eccentricity and
            r.mean_intensity < max_mean_intensity and
            r.major_axis_length > min_major_axis_length and
            r.min_intensity < max_min_intensity)
    ]

def calculate_crop_padding(major_axis_length: float, data: DetectionData) -> int:
    """Calculate appropriate padding based on object size."""
    if major_axis_length < data.small_object_threshold:
        return data.small_object_padding
    elif major_axis_length < data.medium_object_threshold:
        return data.medium_object_padding
    else:
        return data.large_object_padding

def process_regions(regions: List, image_shape: Tuple[int, int], data: DetectionData) -> List[ProcessedRegion]:
    """Process regions: filter, extract data, and save crops."""
    valid_regions = filter_regions(
        regions,
        data.max_eccentricity,
        data.max_mean_intensity,
        data.min_major_axis_length,
        data.max_min_intensity
    )
    if not valid_regions:
        return []
    
    processed_regions: List[ProcessedRegion] = []
    for i, region in enumerate(valid_regions):
        
        row, col = int(region.centroid[0]), int(region.centroid[1])
        padding = calculate_crop_padding(region.major_axis_length, data)
        
        minr = max(0, row - padding)
        minc = max(0, col - padding)
        maxr = min(image_shape[0], row + padding)
        maxc = min(image_shape[1], col + padding)

        region_data = {
            'Area': region.area,
            'MajorAxisLength': region.major_axis_length,
            'MinorAxisLength': region.minor_axis_length,
            'Eccentricity': region.eccentricity,
            'Orientation': region.orientation,
            'EquivDiameter': region.equivalent_diameter,
            'Solidity': region.solidity,
            'Extent': region.extent,
            'MaxIntensity': region.max_intensity,
            'MeanIntensity': region.mean_intensity,
            'MinIntensity': region.min_intensity,
            'Perimeter': region.perimeter
        }

        processed_regions.append(
            ProcessedRegion(
                region_id=i,
                region_extents=(minr, maxr, minc, maxc),
                region_data=region_data
            )
        )
    
    return processed_regions

def create_dataframe(data_list: list, depth_profiles_df: pd.DataFrame) -> pd.DataFrame:
    # Create a combined DataFrame from all processed regions
    if not data_list:
        return pd.DataFrame()
    
    # Create dataframe
    combined_df = pd.DataFrame(data_list)

    # Join with depth profiles
    combined_df = pd.merge(combined_df, depth_profiles_df, on='image_id', how='left')

    # Sort the dataframe by image_id and then by replicate
    combined_df = combined_df.sort_values(by=['image_id', 'replicate'])
    
    # Reorder columns to have FileName first, then metadata, then other data
    cols = ['FileName', 'image_id', 'replicate', 'depth'] + \
            [col for col in combined_df.columns if col not in ['FileName', 'image_id', 'replicate', 'depth']]
    combined_df = combined_df[cols]

    return combined_df

def save_dataframe(combined_df: pd.DataFrame, output_count: int, output_path: str, csv_extension: str, csv_separator: str) -> None:
    """Save detection results to CSV and text files."""
    if combined_df.empty:
        print(f"[DETECTION]: Detection completed. No objects detected.")
        print(f"[DETECTION]: {output_count} vignettes created in {output_path}")
        return

    # Save results
    csv_output_file = os.path.join(Path(output_path).parent, f'object_data{csv_extension}')
    combined_df.to_csv(csv_output_file, sep=csv_separator, index=False)

    print(f"[DETECTION]: Processing completed successfully!")
    print(f"[DETECTION]: {len(combined_df)} objects saved to {csv_output_file}")
    print(f"[DETECTION]: {output_count} vignettes saved to {output_path}")

def process_vignette(mapped_region: MappedImageRegions, output_path: str):
    """
    Prepare vignette data and yield it for each processed region.
    
    This function acts as a generator, yielding the region data, the vignette image,
    and the path to save the vignette for each detected object in a mapped region *AS* 
    the calling method also loops over the regions. In this way, the vignette data does 
    not need to be entirely processed and then stored in memory, but can instead be processed 
    and then saved as it is generated, immediately being released from memory. It's as though 
    the outer loop and this inner loop are connected and running synchronously.
    """
    img_name = Path(mapped_region.source_image_path).stem
    image_id = int(img_name.split('_')[1])
    for region in mapped_region.processed_regions:
        
        # Crop vignette
        minr, maxr, minc, maxc = region.region_extents
        vignette_img = mapped_region.source_image[minr:maxr, minc:maxc]
        
        # Construct vignette path
        vignette_filename = f"{img_name}_vignette_{region.region_id}.png"
        vignette_path = os.path.join(output_path, vignette_filename)
    
        # Add image-specific info to the region data
        region.region_data['FileName'] = vignette_filename
        region.region_data['replicate'] = region.region_id
        region.region_data['image_id'] = image_id
        
        yield region.region_data, vignette_img, vignette_path