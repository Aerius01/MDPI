## source: pyimagesearch - detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning

import numpy as np
import cv2
import os
from tqdm import tqdm
from imutils import paths
from tools.hash.image_hashing import ImageHash

def remove_duplicate_images(dataset_path, remove=False):
	# initialize image hash function
	ih = ImageHash()

	# Grab the paths to all images in our input dataset directory and then initialize our hashes dictionary
	# The list_images function uses os.walk(), which is a recursive function that walks recursively through 
	# the entire directory structure, starting from the root directory specified
	print("\n[DUPLICATES]: computing image hashes...")
	imagePaths = list(paths.list_images(dataset_path))

	# hashes will be a dictionary with the hash as the key and a list of image paths as the value
	hashes = {}

	# Loop over our image paths, generating a hash for each image. Identical images will have the same hash.
	for imagePath in tqdm(imagePaths, desc='[DUPLICATES]'):
		image = cv2.imread(imagePath)
		h = ih.dhash(image)
		
		# Add image path to hash's list, creating new list if the hash doesn't exist. Modifies the dictionary in place.
		hashes.setdefault(h, []).append(imagePath)

	print("[DUPLICATES]: computing image hashes...DONE")
	print('[DUPLICATES]: detecting duplicate images...')

	# Loop over the unique image hashes
	nr_dupliate_img = 0
	for (h, hashedPaths) in tqdm(hashes.items(), desc='[DUPLICATES]'):
		# Check if the hash has more than one image path associated with it in the hashes dictionary
		if len(hashedPaths) > 1:
			# If a dry run, then show duplicates in a horizontally stacked montage, but don't remove them
			if remove is False:
				montage = None
				for p in hashedPaths:
					image = cv2.imread(p)
					image = cv2.resize(image, (500, 500))
					if montage is None:
						montage = image
					else:
						montage = np.hstack([montage, image])
				print("[INFO] hash: {}".format(h))
				cv2.imshow("Montage", montage)
				cv2.waitKey(0)
			else:
				# Loop over all image paths with the same hash except the first, and delete them from the filesystem
				for p in hashedPaths[1:]:
					os.remove(p)
					nr_dupliate_img += 1

	print('[DUPLICATES]: detecting duplicate images...DONE')
	print(f'[DUPLICATES]: {nr_dupliate_img} duplicate images removed')
