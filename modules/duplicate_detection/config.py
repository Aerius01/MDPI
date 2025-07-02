from dataclasses import dataclass

@dataclass
class DuplicateConfig:
    """Configuration for duplicate detection."""
    remove: bool = False
    display_size: tuple = (500, 500)
    show_montages: bool = True 