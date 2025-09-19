from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class PlottingConstants:
    """Centralized plotting constants used by the plotter module."""
    # Plot aesthetics
    FIGSIZE: Tuple[float, float] = (10, 17.5)
    DAY_COLOR: str = 'white'
    NIGHT_COLOR: str = 'grey'
    EDGE_COLOR: str = 'black'
    ALIGN: str = 'edge'
    LEGEND_FONTSIZE: int = 20
    FILE_FORMAT: str = 'png'
    # Imaging
    PIXEL_SIZE_UM: float = 20.9  # micrometers per pixel

    # Column validation
    SINGLE_PROFILE_REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        'project', 'recording_start_date', 'cycle',
        'label', 'depth', 'concentration', 'bin_size'
    ])
    DAY_NIGHT_PROFILE_REQUIRED_COLUMNS: List[str] = field(default_factory=lambda: [
        'project', 'recording_start_date', 'label', 
        'depth', 'concentration', 'bin_size'
    ])


# Global instance for easy access in plotter
PLOTTING_CONSTANTS = PlottingConstants()


