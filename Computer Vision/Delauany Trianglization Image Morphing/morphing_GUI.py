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
import morphing_algorithm as ESSENCE

# Keypoint class
class Keypoint:
	''' the coordinate standard is the same as tkinter coordinate '''
	def __init__(self,x,y,key_id):
		self.key_id = key_id
		self.x = x # x position
		self.y = y # y position

# GUI class
class Labeler_GUI:
	def __init__(self):
		self.root = tk.Tk()
		self.root.title('Image Morphing')

		self.img_1 = 0	# the loaded image 1
		self.img_2 = 0	# the loaded image 2
		self.img_3 = 0	# the output image

		self.keypoints_1 = [] # the list of all keypoints for image 1
		self.keypoints_2 = [] # the list of all keypoints for image 2
		self.avg_keypoints = [] # the list of all average keypoints (avg(img1,img2)) for output image

		# image panels, show the input images: image 1 and 2
		self.panel_1 = tk.Label(self.root)  # initialize image panel 1
		self.panel_2 = tk.Label(self.root)  # initialize image panel 2
		self.panel_1.grid(row=0, column=0)
		self.panel_2.grid(row=0, column=1)

		# load image 1 button
		button1 = tk.Button(self.root, text="Load Image 1")
		button1.bind("<Button-1>", self.load_img_1)
		button1.grid(row=1,column=0)

		# load image 2 button
		button2 = tk.Button(self.root, text="Load Image 2")
		button2.bind("<Button-1>", self.load_img_2)
		button2.grid(row=1,column=1)

		# load keypoints of image 1
		button3 = tk.Button(self.root, text="Load Keypoints of Image 1")
		button3.bind("<Button-1>", self.load_key_1)
		button3.grid(row=2,column=0)

		# load keypoints of image 2
		button4 = tk.Button(self.root, text="Load Keypoints of Image 2")
		button4.bind("<Button-1>", self.load_key_2)
		button4.grid(row=2,column=1)

		# morphing two images
		button5 = tk.Button(self.root, text="Process", command=self.boom_morphing)
		button5.grid(row=3,columnspan=2)
		
		
	# call function from other file (the ESSENCE of this assignment:) to morphe two images and show it on screen
	def boom_morphing(self):
		if len(self.keypoints_1) == len(self.keypoints_2) and len(self.keypoints_1) != 0:
			self.avg_keypoints = ESSENCE.get_average_points(self.keypoints_1, self.keypoints_2) # Get the list of average points
			triangles = ESSENCE.delauany_trianglization(self.avg_keypoints) # trianglize the avg_keypoints
			self.img_3 = ESSENCE.lets_morphing(self.img_1, self.img_2, self.keypoints_1, self.keypoints_2, self.avg_keypoints, triangles)
			cv2.imshow('Output',self.img_3)
		else:
			print('[INFO] keypoints error!')

	# load image from file
	def load_img_1(self,event):
		file_path = filedialog.askopenfilename() # ask user to select the file
		if file_path != '':
			print('[INFO] Image file path: '+file_path)
			img = cv2.imread(file_path,1)
			img = cv2.resize(img, (512,512))
			self.img_1 = img
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
			img = Image.fromarray(img)  # convert image for PIL
			imgtk = ImageTk.PhotoImage(image=img)  # convert image for tkinter
			self.panel_1.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
			self.panel_1.config(image=imgtk)  # show the image

	# load image from file
	def load_img_2(self,event):
		file_path = filedialog.askopenfilename() # ask user to select the file
		if file_path != '':
			print('[INFO] Image file path: '+file_path)
			img = cv2.imread(file_path,1)
			img = cv2.resize(img, (512,512))
			self.img_2 = img
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
			img = Image.fromarray(img)  # convert image for PIL
			imgtk = ImageTk.PhotoImage(image=img)  # convert image for tkinter
			self.panel_2.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
			self.panel_2.config(image=imgtk)  # show the image
			

	# load keypoints 
	def load_key_1(self,event):
		file_path = filedialog.askopenfilename() # ask user the output file name and path
		if file_path != '':
			with open(file_path, 'r', newline='') as csvfile:
				csv_reader = csv.reader(csvfile, delimiter=',')
				for row in csv_reader:
					keypoint = Keypoint(row[0],row[1],-1)
					self.keypoints_1.append(keypoint)
			print('[INFO] Keypoints for image 1 loaded!')

	# load keypoints
	def load_key_2(self,event):
		file_path = filedialog.askopenfilename() # ask user the output file name and path
		if file_path != '':
			with open(file_path, 'r', newline='') as csvfile:
				csv_reader = csv.reader(csvfile, delimiter=',')
				for row in csv_reader:
					keypoint = Keypoint(row[0],row[1],-1)
					self.keypoints_2.append(keypoint)
			print('[INFO] Keypoints for image 2 loaded!')


# start the GUI
labeler_gui = Labeler_GUI()
labeler_gui.root.mainloop()