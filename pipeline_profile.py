import datetime
import os
from pathlib import Path
from typing import Union, List

class Profile:
    """A class to hold profile information for a pipeline run."""

    def __init__(self, directory_path: Union[str, os.PathLike], output_folder: Union[str, os.PathLike]):
        """
        Initializes the Profile object by parsing a directory path and an image filename within it.

        Args:
            directory_path: Path to the directory containing the images.
            output_folder: Path to the directory where output files will be saved.
        """
        self.directory_path = Path(directory_path).resolve()
        self.output_folder = Path(output_folder).resolve()
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.deduplicated_imgs: List[str] = []
        
        self._parse_path_metadata()
        self._parse_file_metadata()

    def _parse_path_metadata(self):
        """Parses the directory path to extract project, date, cycle, and location."""
        try:
            p = self.directory_path
            self.location = p.name
            self.cycle = p.parent.name
            date_str = p.parent.parent.name
            self.project = p.parent.parent.parent.name
        except IndexError:
            raise ValueError(f"Directory path '{self.directory_path}' is not in the expected format '.../project/date/cycle/location'")

        try:
            self.recording_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            raise ValueError(f"Date '{date_str}' from path is not in 'YYYYMMDD' format.")

    def _parse_file_metadata(self):
        """Parses an image filename to get the replicate number and recording time."""
        p = self.directory_path
        
        # Define common image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
        filenames = sorted([f for f in os.listdir(p) if os.path.splitext(f)[1].lower() in image_extensions])
        if not filenames:
            raise ValueError(f"No image files found in directory '{p}'")

        self.raw_imgs = [str(p / filename) for filename in filenames]

        # Grab a template filename by selecting the first file in the directory.
        image_filename = filenames[0]

        try:
            base_filename, _ = os.path.splitext(image_filename)
            filename_parts = base_filename.split("_")

            if len(filename_parts) < 4:
                raise ValueError("Filename does not have enough parts separated by '_'")

            self.replicate = int(filename_parts[-1])
            date_from_filename_str = filename_parts[-3]
            time_str = filename_parts[-2]

            date_from_filename = datetime.datetime.strptime(date_from_filename_str, "%Y%m%d").date()

            if date_from_filename != self.recording_date:
                raise ValueError(f"Date in filename '{date_from_filename}' does not match date in path '{self.recording_date}'.")

            if len(time_str) != 9:
                raise ValueError(f"Time part of filename '{time_str}' has incorrect length. Expected 9 digits for 'HHMMSSmmm'.")

            # strptime's %f directive expects microseconds (6 digits).
            # Convert the millisecond part of the time string to microseconds.
            time_str_for_strptime = time_str[:6] + f"{int(time_str[6:]) * 1000:06d}"
            self.recording_time = datetime.datetime.strptime(time_str_for_strptime, "%H%M%S%f").time()

        except (IndexError, ValueError) as e:
            if isinstance(e, ValueError) and ("does not match" in str(e) or "incorrect length" in str(e) or "enough parts" in str(e)):
                raise
            raise ValueError(f"Could not parse info from filename '{image_filename}'. Expected '..._YYYYMMDD_HHMMSSmmm_replicate.ext' format. Error: {e}")

    def set_deduplicated_imgs(self, removed_paths: List[str]):
        """
        Sets the deduplicated_imgs attribute by removing the specified paths from raw_imgs.

        Args:
            removed_paths: A list of file paths that have been removed.
        """
        remaining_files = sorted(list(set(self.raw_imgs) - set(removed_paths)))
        self.deduplicated_imgs = remaining_files 