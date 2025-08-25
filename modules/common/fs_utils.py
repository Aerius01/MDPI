"""
Filesystem utilities shared across modules.
"""

from pathlib import Path


def ensure_dir(output_path: str) -> Path:
    """Validate and create output path if needed, returning a Path object."""
    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


