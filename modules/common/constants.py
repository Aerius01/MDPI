from dataclasses import dataclass

@dataclass(frozen=True)
class ProcessingConstants:
    """Centralized processing constants grouped by module for readability.

    The attributes are organized in the same order as the pipeline:
    depth profiling → flatfielding → object detection → object classification.
    Shared items come first. Names are preserved to avoid breaking imports.
    """

    # ── Shared/common (used across modules) ────────────────────────────────────
    BATCH_SIZE: int = 10  # How many images to process at a given time?
    CSV_EXTENSION: str = '.csv'

# Global instance for easy access
CONSTANTS = ProcessingConstants()