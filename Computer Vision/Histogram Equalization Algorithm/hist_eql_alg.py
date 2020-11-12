import cv2
import numpy as np

# The histogram equalization algorithm
def hist_eql_alg(img):
	''' Input: a color image
		Return: the processed image after histogram equalization '''
	HEIGHT = len(img) # the height of the image
	WIDTH = len(img[0]) # the width of the image

	OUTPUT = np.array(img) # the output image (same size as the imput image)

	for channel in range(4):
		''' For each channel, apply this algorithm separately '''

		hist = np.zeros([256]) # Histogram
		
		# Get the histogram of for each value in [0,255] 
		for y in range(HEIGHT):
			for x in range(WIDTH):
				index = img[y][x][channel]
				hist[index] = hist[index] + 1

		# Compute the cumulative histogram
		for index in range(1,256):
			hist[index] = hist[index] + hist[index-1]

		# Divide by the number of pixels
		hist = hist/(WIDTH*HEIGHT)

		# Map new value to the output image
		for y in range(HEIGHT):
			for x in range(WIDTH):
				index = int(img[y][x][channel])
				OUTPUT[y][x][channel] = 255*(hist[index])

	return OUTPUT