## source: pyimagesearch - detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning

import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm
from imutils import paths
from tools.hash.image_hashing import ImageHash

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1, help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())

# initialize image hash function
ih = ImageHash()

# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = {}
# loop over our image paths
for imagePath in tqdm(imagePaths):
	# load the input image and compute the hash
	image = cv2.imread(imagePath)
	h = ih.dhash(image)
	# grab all image paths with that hash, add the current image
	# path to it, and store the list back in the hashes dictionary
	p = hashes.get(h, [])
	p.append(imagePath)
	hashes[h] = p
print("[INFO] computing image hashes...DONE")

print('[INFO] detecting duplicate images...')
nr_dupliate_img = 0

# loop over the image hashes
for (h, hashedPaths) in tqdm(hashes.items()):
	# check to see if there is more than one image with the same hash
	if len(hashedPaths) > 1:
		# check to see if this is a dry run
		if args["remove"] <= 0:
			# initialize a montage to store all images with the same
			# hash
			montage = None
			# loop over all image paths with the same hash
			for p in hashedPaths:
				# load the input image and resize it to a fixed width
				# and heightG
				image = cv2.imread(p)
				image = cv2.resize(image, (500, 500))
				# if our montage is None, initialize it
				if montage is None:
					montage = image
				# otherwise, horizontally stack the images
				else:
					montage = np.hstack([montage, image])
			# show the montage for the hash
			print("[INFO] hash: {}".format(h))
			cv2.imshow("Montage", montage)
			cv2.waitKey(0)
			# otherwise, we'll be removing the duplicate images
		else:
			# loop over all image paths with the same hash *except*
			# for the first image in the list (since we want to keep
			# one, and only one, of the duplicate images)
			for p in hashedPaths[1:]:
				os.remove(p)
				# update img removal count
				nr_dupliate_img += 1
print('[INFO] detecting duplicate images...DONE')
print(f'[INFO] {nr_dupliate_img} duplicate images removed')
