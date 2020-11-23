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

# Keypoint class
class Keypoint:
	''' the coordinate standard is the same as tkinter coordinate '''
	def __init__(self,x,y):
		self.x = x # x position
		self.y = y # y position

# GUI class
class Labeler_GUI:
	def __init__(self):
		self.root = tk.Tk()
		self.root.title('Image Keypoints Labeler')

		self.img = 0	# the loaded image

		self.keypoints = [] # the list of all keypoints

		# image panel, show the image
		self.panel = tk.Label(self.root)  # initialize image panel
		self.panel.pack(padx=0, pady=0)
		self.panel.bind('<Button-1>', self.get_click_position)

		# load image button
		button1 = tk.Button(self.root, text="Load Image File", command=self.load_img)
		button1.pack()

		# save keypoints to file button
		button2 = tk.Button(self.root, text="Save Keypoints", command=self.save_key)
		button2.pack()

	def get_click_position(self, event):
		print('[INFO] Click on '+str(event.x-4)+','+str(event.y-4)) # -4 pixels to offset the margin space!!
		self.keypoints.append(Keypoint(event.x-4, event.y-4)) # record into the list
		''' display a label on that keypoint '''
		label = tk.Label(self.root, text=str(len(self.keypoints)), font=("Helvetica", 8), fg="red")
		label.place(x=event.x, y=event.y)

	# load image from file
	def load_img(self):
		file_path = filedialog.askopenfilename() # ask user to select the file
		if file_path != '':
			print('[INFO] Image file path: '+file_path)
			self.img = cv2.imread(file_path,1)
			self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
			self.img = cv2.resize(self.img, (512,512))
			self.img = Image.fromarray(self.img)  # convert image for PIL
			imgtk = ImageTk.PhotoImage(image=self.img)  # convert image for tkinter
			self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
			self.panel.config(image=imgtk)  # show the image
			

	# save keypoints as a .csv file
	def save_key(self):
		file_path = filedialog.asksaveasfilename() # ask user the output file name and path
		if file_path != '':
			with open(file_path, 'w', newline='') as csvfile:
				csv_writer = csv.writer(csvfile, delimiter=',')
				for keypoint in self.keypoints:
					csv_writer.writerow([keypoint.x, keypoint.y])
			print('[INFO] Saved!')


# start the GUI
labeler_gui = Labeler_GUI()
labeler_gui.root.mainloop()