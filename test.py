import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import imgutils
import photometric_alignment
import pyramid
import se3utils


# Parse command-line arguments
def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('-datapath', help='Path to a TUM RGB-D Odometry benchmark sequence', \
		required=True)
	parser.add_argument('-startFrameRGB', help='Filename (sans the .png extension) of the first \
		RGB frame to be processed', required=True)
	parser.add_argument('-startFrameDepth', help='Filename (sans the .png extension) of the first \
		depth frame to be processed', required=True)
	parser.add_argument('-endFrameRGB', help='Filename (sans the .png extension) of the last \
		RGB frame to be processed')
	parser.add_argument('-endFrameDepth', help='Filename (sans the .png extension) of the last \
		depth frame to be processed')
	parser.add_argument('-numPyramidLevels', help='Number of levels used in the pyramid', default=3)
	parser.add_argument('-stepsize', help='Step-size parameter, for gradient descent', default=1e-6)
	parser.add_argument('-numIters', help='Default number of iterations to run each optimization \
		routine for', default=50)
	parser.add_argument('-tol', help='Tolerance parameter for gradient-based optimization. \
		Specifies the amount by which loss must change across successive iterations', default=1e-8)


	args = parser.parse_args()

	return args


# Main method
def main(args):
	
	img_gray_prev = cv2.imread(os.path.join(args.datapath, 'rgb', args.startFrameRGB + '.png'), cv2.IMREAD_GRAYSCALE)
	img_depth_prev = cv2.imread(os.path.join(args.datapath, 'depth', args.startFrameDepth + '.png'), cv2.IMREAD_GRAYSCALE)
	img_gray_cur = cv2.imread(os.path.join(args.datapath, 'rgb', args.endFrameRGB + '.png'), cv2.IMREAD_GRAYSCALE)
	img_depth_cur = cv2.imread(os.path.join(args.datapath, 'depth', args.endFrameDepth + '.png'), cv2.IMREAD_GRAYSCALE)

	# Convert the intensity images to float
	img_gray_prev = imgutils.im2float(img_gray_prev)
	img_gray_cur = imgutils.im2float(img_gray_cur)
	# print(img_gray_prev.shape, img_depth_prev.shape, img_gray_cur.shape, img_depth_cur.shape)
	
	# Use default camera intrinsics, for now
	f = 525.0
	cx = 319.5
	cy = 239.5
	scaling_factor = 5000

	# Construct a downsampled pyramid using the specified number of pyramid levels
	pyramid_gray, pyramid_depth, pyramid_intrinsics = pyramid.buildPyramid(img_gray_prev, \
		img_depth_prev, num_levels=args.numPyramidLevels, focal_length=f, cx=cx, cy=cy)

	# Compute residuals
	K = dict()
	K['f'] = f
	K['cx'] = cx
	K['cy'] = cy
	K['scaling_factor'] = scaling_factor
	xi_init = np.zeros((6,1))
	# residuals, cache_point3d = photometric_alignment.computeResiduals(img_gray_prev, img_depth_prev, \
	# 	img_gray_cur, K, xi_init)

	# # Test image gradient computation
	# grad_ix, grad_iy = photometric_alignment.computeImageGradients(img_gray_prev)
	# cv2.imshow('img', img_gray_prev)
	# cv2.imshow('grad_x', grad_ix)
	# cv2.imshow('grad_y', grad_iy)
	# cv2.waitKey(0)

	# # Test Jacobian computation
	# J = photometric_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, \
	# 	K, xi_init, residuals, cache_point3d)

	# Simple gradient descent test
	stepsize = 1e-6
	max_iters = 100
	tol = 1e-8
	err_prev = 1e24
	for it in range(max_iters):
		residuals, cache_point3d = photometric_alignment.computeResiduals(img_gray_prev, \
			img_depth_prev, img_gray_cur, K, xi_init)
		J = photometric_alignment.computeJacobian(img_gray_prev, img_depth_prev, img_gray_cur, \
			K, xi_init, residuals, cache_point3d)
		# Normalize the error and the jacobian
		err_cur = 0.5 * (1 / (img_gray_cur.shape[0]*img_gray_cur.shape[1])) * np.sum(np.abs(residuals))
		grad = (1 / (img_gray_cur.shape[0]*img_gray_cur.shape[1])) * np.reshape(np.sum(J, axis=(0,1)).T, (6,1))
		print('Error: ', err_cur)
		print('Jacobian: ', np.sum(J, axis=(0,1)))
		xi_init += stepsize * grad
		if np.abs(err_prev - err_cur) < tol:
			break
		err_prev = err_cur

	
	# fig, ax = plt.subplots(2, 2)
	# ax[0, 0].imshow(img_gray_prev, cmap='gray')
	# ax[0, 0].set_title('RGB image (current frame)')
	# ax[0, 1].imshow(img_depth_prev, cmap='gray')
	# ax[0, 1].set_title('Depth image (current frame)')
	# ax[1, 0].imshow(img_gray_cur, cmap='gray')
	# ax[1, 0].set_title('RGB image (next frame)')
	# ax[1, 1].imshow(img_depth_cur, cmap='gray')
	# ax[1, 1].set_title('Depth image (next frame)')
	# plt.show()


if __name__ == '__main__':
	args = parse_args()
	main(args)
