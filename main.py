from duplicate_image_removal import remove_duplicate_images
from modified_renaming_profiles import rename_profiles
from flatfielding_profiles import flatfielding_profiles
from dateutil import relativedelta

# Physical camera capture rate is 2.4 Hz due to data transfer from network connection to computer
# if this is not expressed as microseconds, the comparison will not be sensitive enough to capture the time difference
TIMESTEP = relativedelta.relativedelta(microseconds=1/2.4*1000000)

dataset_path = "./profiles"
depth_profiles_path = "./depth_profiles"
flatfield_profiles_path = "./flatfield_profiles"

# Self-explanatory
remove_duplicate_images(dataset_path, remove=False)

# Add the depth value to the image file names
rename_profiles(dataset_path, depth_profiles_path, TIMESTEP)

# Flatfield the images
flatfielding_profiles(depth_profiles_path, flatfield_profiles_path)