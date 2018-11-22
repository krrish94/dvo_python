"""
Utilities for image processing convenience
"""

import cv2


def im2float(img):
	# Convert input image to normalized float
	return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
