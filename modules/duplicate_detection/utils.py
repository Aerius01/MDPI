from typing import List, Callable, Optional
import os
from imutils import paths


def get_image_group_from_folder(image_folder: str, sort_key: Optional[Callable] = None) -> List[str]:
    """Get list of image paths from a folder with optional custom sorting."""
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")

    image_group = list(paths.list_images(image_folder))
    if not image_group:
        raise ValueError(f"No images found in folder: {image_folder}")

    if sort_key:
        image_group = sorted(image_group, key=sort_key)
    else:
        image_group = sorted(image_group)

    return image_group


