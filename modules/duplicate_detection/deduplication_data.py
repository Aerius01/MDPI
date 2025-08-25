from dataclasses import dataclass
import argparse
from modules.duplicate_detection.utils import get_image_group_from_folder

@dataclass
class DeduplicationData:
    image_paths: list
    display_size: tuple
    remove: bool
    show_montages: bool

def process_arguments(args: argparse.Namespace) -> DeduplicationData:
    # Handle montage display logic
    show_montages = args.show_montages and not args.no_montages
    
    # Get image paths from input directory
    print(f"[DUPLICATES]: Loading images from {args.input}")
    image_paths = get_image_group_from_folder(args.input)
    print(f"[DUPLICATES]: Found {len(image_paths)} images")
    
    return DeduplicationData(
        image_paths=image_paths,
        display_size=args.display_size,
        remove=args.remove,
        show_montages=show_montages
    )