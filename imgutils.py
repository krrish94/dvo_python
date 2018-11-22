"""
Utilities
"""

import cv2
import numpy as np


def im2float(img):
	# Convert input image to normalized float
	return cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)


# Interpolate an intensity value bilinearly (useful when a warped point is not integral)
# Translated from: https://github.com/muskie82/simple_dvo/blob/master/src/util.cpp
def bilinear_interpolation(img, x, y, width, height):

	# Consider the pixel as invalid, to begin with
	valid = np.nan

	# Get the four corner coordinates for the current floating point values x, y
	x0 = np.floor(x).astype(np.uint8)
	y0 = np.floor(y).astype(np.uint8)
	x1 = x0 + 1
	y1 = y0 + 1

	# Compute weights for each corner location, inversely proportional to the distance
	x1_weight = x - x0
	y1_weight = y - y0
	x0_weight = 1 - x1_weight
	y0_weight = 1 - y1_weight

	# Check if the warped points lie within the image
	if x0 < 0 or x0 >= width:
		x0_weight = 0
	if x1 < 0 or x1 >= width:
		x0_weight = 0
	if y0 < 0 or y0 >= height:
		y0_weight = 0
	if y1 < 0 or y1 >= height:
		y1_weight = 0

	# Bilinear weights
	w00 = x0_weight * y0_weight
	w10 = x1_weight * y0_weight
	w01 = x0_weight * y1_weight
	w11 = x1_weight * y1_weight

	# Bilinearly interpolate intensities
	sum_weights = w00 + w10 + w01 + w11
	total = 0
	if w00 > 0:
		total += img.item((y0, x0)) * w00
	if w01 > 0:
		total += img.item((y1, x0)) * w01
	if w10 > 0:
		total += img.item((y0, x1)) * w10
	if w11 > 0:
		total += img.item((y1, x1)) * w11

	if sum_weights > 0:
		valid = total / sum_weights

	return valid
