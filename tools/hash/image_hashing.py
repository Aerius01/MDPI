import cv2


class ImageHash:
    @staticmethod
    def dhash(image, hashSize=8):
        # convert the image to grayscale and resize the grayscale image,
        # adding a single column (width) so we can compute the horizontal
        # gradient
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hashSize + 1, hashSize))
        # compute the (relative) horizontal gradient between adjacent
        # column pixels
        diff = resized[:, 1:] > resized[:, :-1]
        # convert the difference image to a hash and return it
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
