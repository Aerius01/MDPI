from pathlib import Path
from typing import Dict
from modules.common.parser import _parse_path_metadata, _list_image_filenames, _build_image_paths, _parse_hhmmssmmm


def parse_flatfield_metadata_from_directory(directory_path: Path) -> Dict:
    """
    Parses metadata from a directory containing flat-fielded images.

    This function is designed to work with the output of the flatfielding module.
    It extracts metadata from both the directory path and the image filenames.
    """
    path_metadata = _parse_path_metadata(directory_path.parent)
    filenames = _list_image_filenames(directory_path)

    # The time is the same for all files in a batch, so we can parse it from the first one.
    time_str = filenames[0].split('_')[0]
    recording_start_time = _parse_hhmmssmmm(time_str)

    # To get total replicates, use the last (highest numbered) filename
    # Filenames are sorted, so the last one has the highest replicate ID
    last_filename = filenames[-1]
    last_replicate_id = int(last_filename.split('_')[1].split('.')[0])
    
    # Determine if numbering is 0-based or 1-based by checking the first image (MDPI-dependent)
    first_filename = filenames[0]
    first_replicate_id = int(first_filename.split('_')[1].split('.')[0])
    
    # Calculate total_replicates: add 1 for 0-based numbering, use as-is for 1-based
    total_replicates = last_replicate_id + (1 if first_replicate_id == 0 else 0)

    raw_img_paths = _build_image_paths(directory_path, filenames)

    file_metadata = {
        "recording_start_time": recording_start_time,
        "total_replicates": total_replicates,
        "flatfield_img_paths": raw_img_paths,
    }

    return {**path_metadata, **file_metadata}


