"""
Functions to compute photometric error residuals and jacobians
"""

import numpy as np

import imgutils
import se3utils


# Takes in an intensity image and a registered depth image, and outputs a pointcloud
# Intrinsics must be provided, else we use the TUM RGB-D benchmark defaults.
# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
def rgbd_to_pointcloud(gray, depth, focal_length, cx, cy, scaling_factor):
	pointcloud = []
	for v in range(gray.shape[1]):
		for u in range(gray.shape[0]):
			intensity = gray.item((u, v))
			Z = depth.item((u, v)) / scaling_factor
			if Z == 0:
				continue
			X = (u - cx) * Z / focal_length
			Y = (v - cy) * Z / focal_length
			pointcloud.append((X, Y, Z, intensity))
	return pointcloud


# Compute photometric error (i.e., the residuals)
def computeResiduals(gray_prev, depth_prev, gray_cur, K, xi):
	"""
	Computes the image alignment error (residuals). Takes in the previous intensity image and 
	first backprojects it to 3D to obtain a pointcloud. This pointcloud is then rotated by an 
	SE(3) transform "xi", and then projected down to the current image. After this step, an 
	intensity interpolation step is performed and we compute the error between the projected 
	image and the actual current intensity image.
	"""

	width = gray_cur.shape[1]
	height = gray_cur.shape[0]
	residuals = np.zeros(gray_cur.shape, dtype = np.float32)

	# # Backproject an image to 3D to obtain a pointcloud
	# pointcloud = rgbd_to_pointcloud(gray_prev, depth_prev, focal_length=K['f'], cx=K['cx'], 
	# 	cy=K['cy'], scaling_factor = K['scaling_factor'])

	one_by_f = 1. / K['f']

	# TODO: Replace this with the SE(3) Exponential map for xi
	T = se3utils.SE3_Exp(xi)

	K_mat = np.asarray([[K['f'], 0, K['cx']], [0, K['f'], K['cy']], [0, 0, 1]])

	# Warp each point in the previous image, to the current image
	for v in range(gray_prev.shape[0]):
		for u in range(gray_prev.shape[1]):
			intensity_prev = gray_prev.item((v,u))
			Z = depth_prev.item((v,u)) / K['scaling_factor']
			if Z <= 0:
				continue
			Y = one_by_f * Z * (v - K['cy'])
			X = one_by_f * Z * (u - K['cx'])
			# Transform the 3D point
			point_3d = np.dot(T[0:3,0:3], np.asarray([X, Y, Z])) + T[0:3,3]
			point_3d = np.reshape(point_3d, (3,1))
			# Project it down to 2D
			point_2d_warped = np.dot(K_mat, point_3d)
			px = point_2d_warped[0] / point_2d_warped[2]
			py = point_2d_warped[1] / point_2d_warped[2]

			# Interpolate the intensity value bilinearly
			intensity_warped = imgutils.bilinear_interpolation(gray_cur, px[0], py[0], width, height)

			# If the pixel is valid (i.e., interpolation return a non-NaN value), compute residual
			if not np.isnan(intensity_warped):
				residuals.itemset((v, u), intensity_prev - intensity_warped)

	return residuals


# Function to compute image gradients (used in Jacobian computation)
def computeImageGradients(img):
	"""
	We use a simple form for the image gradient. For instance, a gradient along the X-direction 
	at location (y, x) is computed as I(y, x+1) - I(y, x-1).
	"""
	gradX = np.zeros(img.shape, dtype = np.float32)
	gradY = np.zeros(img.shape, dtype = np.float32)

	width = img.shape[1]
	height = img.shape[0]

	# Exploit the fact that we can perform matrix operations on images, to compute gradients quicker
	gradX[:, 1:width-1] = img[:, 2:] - img[:,0:width-2]
	gradY[1:height-1, :] = img[2:, :] - img[:height-2, :]

	return gradX, gradY
