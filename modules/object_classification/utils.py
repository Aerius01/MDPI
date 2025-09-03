from pathlib import Path
from typing import Dict
from modules.common.parser import _parse_path_metadata, _list_image_filenames, _build_image_paths, _parse_hhmmssmmm


def parse_vignette_file_metadata(directory_path: Path) -> Dict:
    filenames = _list_image_filenames(directory_path)
    raw_img_paths = _build_image_paths(directory_path, filenames)

    last_image_filename = filenames[-1]
    filename_parts = last_image_filename.split('_')
    time_str = filename_parts[0]
    recording_start_time = _parse_hhmmssmmm(time_str)

    # Determine if numbering is 0-based or 1-based by checking the first image (MDPI-dependent)
    first_filename = filenames[0]
    first_parts = Path(first_filename).stem.split("_")
    first_replicate_id = int(first_parts[-1])
    
    # Calculate total_replicates: add 1 for 0-based numbering, use as-is for 1-based
    total_replicates = int(filename_parts[-1]) + (1 if first_replicate_id == 0 else 0)
        
    return {
        "total_replicates": total_replicates,
        "recording_start_time": recording_start_time,
        "raw_img_paths": raw_img_paths
    }


def parse_vignette_metadata(directory_path: Path) -> Dict:
    """Parses directory path and vignette filenames to extract all metadata."""
    path_metadata = _parse_path_metadata(directory_path.parent)
    file_metadata = parse_vignette_file_metadata(directory_path)
    return {**path_metadata, **file_metadata}


