from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingConstants:
    """Centralized processing constants"""

    # ── Shared/common (used across modules) ────────────────────────────────────
    BATCH_SIZE: int = 10  # How many images to process at a given time?
    CSV_EXTENSION: str = '.csv'
    CSV_SEPARATOR: str = ';'
    FLATFIELD_DIR_NAME: str = "flatfielded_images"
    VIGNETTES_DIR_NAME: str = "vignettes"

# Global instance for easy access
CONSTANTS = ProcessingConstants()