from duplicate_image_removal import remove_duplicate_images
from depth_profiler import profile_depths
from flatfielding_profiles import flatfielding_profiles
from dateutil import relativedelta
from object_detection import detect_objects

# Physical camera capture rate is 2.4 Hz due to data transfer from network connection to computer
# if this is not expressed as microseconds, the comparison will not be sensitive enough to capture the time difference
TIMESTEP = relativedelta.relativedelta(microseconds=1/2.4*1000000)

dataset_path = "./profiles"
depth_profiles_path = "./depth_profiles"
flatfield_profiles_path = "./flatfield_profiles"
object_detection_path = "./vignettes"

# # Self-explanatory
# remove_duplicate_images(dataset_path, remove=False)

# # Add the depth value to the image file names
# profile_depths(dataset_path, depth_profiles_path, TIMESTEP)

# # Flatfield the images
# flatfielding_profiles(depth_profiles_path, flatfield_profiles_path)

# Detect objects in the images
detect_objects(flatfield_profiles_path, object_detection_path)