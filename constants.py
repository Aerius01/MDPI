import re
from dateutil import relativedelta

# Pre-compile regex for efficiency - matches datetime pattern in image filenames
BASE_FILENAME_PATTERN = re.compile(r'(\d{8}_\d{6}\d{3})_(\d+)\.') 

# Physical camera capture rate is 2.4 Hz due to data transfer from network connection to computer
# if this is not expressed as microseconds, the comparison will not be sensitive enough to capture the time difference
TIMESTEP = relativedelta.relativedelta(microseconds=1/2.4*1000000)

# Process images in batches to reduce memory usage
BATCH_SIZE = 10  

# Flatfielding
NORMALIZATION_FACTOR = 235
