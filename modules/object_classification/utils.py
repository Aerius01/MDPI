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

    total_replicates = int(filename_parts[1])
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


