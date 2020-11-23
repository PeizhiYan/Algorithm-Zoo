from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import argparse
import datetime
import cv2
import os
from tkinter import filedialog
import time
import csv
from scipy.spatial import Delaunay

# Keypoint class
class Keypoint:
	''' the coordinate standard is the same as tkinter coordinate '''
	def __init__(self,x,y,key_id):
		self.key_id = key_id
		self.x = x # x position
		self.y = y # y position

# calculate the average points (x,y): x=(x1+x2)/2, y=(y1+y2)/2
def get_average_points(keypoints_1, keypoints_2):
	output = []
	for i in range(len(keypoints_1)):
		x1 = int(keypoints_1[i].x)
		y1 = int(keypoints_1[i].y)
		x2 = int(keypoints_2[i].x)
		y2 = int(keypoints_2[i].y)
		x = int((x1+x2)/2)
		y = int((y1+y2)/2)
		output.append(Keypoint(x,y,i))
	return output

# trangulize the points using Delauany Triianglization algorithm
# accept a list keypoints
# return a set of triangles
def delauany_trianglization(keypoints):
	points = []
	for keypoint in keypoints:
		points.append([keypoint.x, keypoint.y])
	points = np.array(points)
	return Delaunay(points).simplices.copy() # the list of triangles

# the main morphing algorithm
def lets_morphing(img1, img2, keypoints_1, keypoints_2, avg_keypoints, triangles):
	'''
	input img should be returned from cv2.imread('file')!!
	keypoints' elements should be Keypoint objects
	output is the morphed image array
	'''

	for keypoint in avg_keypoints:
		print(keypoint.x)

	# Output image 1 is set to white
	img_out_1 = 255 * np.ones(img1.shape, dtype = img1.dtype)
	morph_2_images(img1, img_out_1, keypoints_1, avg_keypoints, triangles)

	# Output image 2 is set to white
	img_out_2 = 255 * np.ones(img2.shape, dtype = img2.dtype)
	morph_2_images(img2, img_out_2, keypoints_2, avg_keypoints, triangles)

	# Output image is set to white
	BLENDING_COEFFICIENT = 0.002 # to blend two images!!
	img_out = (BLENDING_COEFFICIENT*img_out_1 + BLENDING_COEFFICIENT*img_out_2)

	return img_out # output image

def morph_2_images(img1, img2, keypoints_1, keypoints_2, triangles):

	# traverse all the trangles
	for i in range(len(triangles)):

		# triangle in first image
		tri1 = np.float32([[
			[int(keypoints_1[triangles[i][0]].x), int(keypoints_1[triangles[i][0]].y)], 
			[int(keypoints_1[triangles[i][1]].x), int(keypoints_1[triangles[i][1]].y)], 
			[int(keypoints_1[triangles[i][2]].x), int(keypoints_1[triangles[i][2]].y)]]])

		# triangle in output image
		tri2 = np.float32([[
			[int(keypoints_2[triangles[i][0]].x), int(keypoints_2[triangles[i][0]].y)], 
			[int(keypoints_2[triangles[i][1]].x), int(keypoints_2[triangles[i][1]].y)], 
			[int(keypoints_2[triangles[i][2]].x), int(keypoints_2[triangles[i][2]].y)]]])

		# Find bounding box. 
		r1 = cv2.boundingRect(tri1)
		r2 = cv2.boundingRect(tri2)

		# Offset points by left top corner of the 
		# respective rectangles
		tri1Cropped = []
		tri2Cropped = []

		for i in range(0, 3):
			tri1Cropped.append(((tri1[0][i][0] - r1[0]),(tri1[0][i][1] - r1[1])))
			tri2Cropped.append(((tri2[0][i][0] - r2[0]),(tri2[0][i][1] - r2[1])))

		# Apply warpImage to small rectangular patches
		img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

		# Given a pair of triangles, find the affine transform.
		warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))

		# Apply the Affine Transform just found to the src image
		img2Cropped = cv2.warpAffine(img1Cropped, warpMat, (r2[2], r2[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

		# Get mask by filling triangle
		mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
		cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

		# Apply mask to cropped region
		img2Cropped = img2Cropped * mask
		
		# Copy triangular region of the rectangular patch to the output image
		img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
		
		img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped
		

