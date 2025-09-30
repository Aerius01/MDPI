import cv2


def rotate_image(image, angle):
    rows, cols = image.shape
    rotM = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rot = cv2.warpAffine(image, rotM, (cols, rows))
    return rot
