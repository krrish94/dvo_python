import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import imgutils
import pyramid


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

	args = parser.parse_args()

	return args


# Main method
def main(args):
	
	img_gray_cur = cv2.imread(os.path.join(args.datapath, 'rgb', args.startFrameRGB + '.png'), cv2.IMREAD_GRAYSCALE)
	img_depth_cur = cv2.imread(os.path.join(args.datapath, 'depth', args.startFrameDepth + '.png'), cv2.IMREAD_GRAYSCALE)
	img_gray_next = cv2.imread(os.path.join(args.datapath, 'rgb', args.endFrameRGB + '.png'), cv2.IMREAD_GRAYSCALE)
	img_depth_next = cv2.imread(os.path.join(args.datapath, 'depth', args.endFrameDepth + '.png'), cv2.IMREAD_GRAYSCALE)

	# Convert the intensity images to float
	img_gray_cur = imgutils.im2float(img_gray_cur)
	img_gray_next = imgutils.im2float(img_gray_next)
	# print(img_gray_cur.shape, img_depth_cur.shape, img_gray_next.shape, img_depth_next.shape)
	
	# Construct a downsampled pyramid using the specified number of pyramid levels
	pyramid_gray, pyramid_depth = pyramid.buildPyramid(img_gray_cur, img_depth_cur, num_levels = args.numPyramidLevels)
	
	# fig, ax = plt.subplots(2, 2)
	# ax[0, 0].imshow(img_gray_cur, cmap='gray')
	# ax[0, 0].set_title('RGB image (current frame)')
	# ax[0, 1].imshow(img_depth_cur, cmap='gray')
	# ax[0, 1].set_title('Depth image (current frame)')
	# ax[1, 0].imshow(img_gray_next, cmap='gray')
	# ax[1, 0].set_title('RGB image (next frame)')
	# ax[1, 1].imshow(img_depth_next, cmap='gray')
	# ax[1, 1].set_title('Depth image (next frame)')
	# plt.show()


if __name__ == '__main__':
	args = parse_args()
	main(args)
