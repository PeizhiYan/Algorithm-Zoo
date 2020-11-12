from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
import argparse
import datetime
import cv2
import os
from tkinter import filedialog
import time
from hist_eql_alg import hist_eql_alg # the histogram equalization algorithm

class GUI:
	def __init__(self):
		self.webcam = cv2.VideoCapture(0) # capture video frames, 0 is your default video camera
		self.video_buffer = None # Be used to load a video

		self.output_path = ''  # store output path
		self.current_image = None  # current image

		self.root = tk.Tk()  # initialize root window
		self.root.title('Histogram Equalication w/ GUI')  # set window title
		self.root.protocol('WM_DELETE_WINDOW', self.destructor) # self.destructor function gets fired when the window is closed

		self.panel = tk.Label(self.root)  # initialize image panel
		self.panel.pack(padx=10, pady=10)

		self.video_out = None # Save video to disk
		self.record_video_status = 0 # 0: recorder is off, 1: recorder is on

		self.status = 0 # 0: Nothing, 1: Webcam, 2: Display an image, 3: Display a video,

		label1 = tk.Label(self.root, text='Video Related Functions', fg='red')
		label1.pack(fill='both', expand=True)

		# open webcam button
		self.btn_1 = tk.Button(self.root, text='Open Webcam', command=self.open_webcam, fg='blue')
		self.btn_1.pack(fill='both', expand=True)

		# start/stop to capture video
		self.btn_3 = tk.Button(self.root, text='Start Record Video', command=self.record_video)
		self.btn_3.pack(fill='both', expand=True)

		# open video from file button
		btn_4 = tk.Button(self.root, text='Open Video From File', command=self.open_video)
		btn_4.pack(fill='both', expand=True)

		label2 = tk.Label(self.root, text='Snapshot the image/video and save', fg='green')
		label2.pack(fill='both', expand=True)

		# take snapshot button
		btn = tk.Button(self.root, text='Save Snapshot', command=self.take_snapshot)
		btn.pack(fill='both', expand=True)

		label3 = tk.Label(self.root, text='Still Image Related Functions', fg='blue')
		label3.pack(fill='both', expand=True)

		# open image from file button
		btn_2 = tk.Button(self.root, text='Open Image From File', command=self.open_image)
		btn_2.pack(fill='both', expand=True)

		# histogram equalization button
		btn_5 = tk.Button(self.root, text='Apply Histogram Equalization', command=self.hist_eql)
		btn_5.pack(fill='both', expand=True)

	# Histogram equalization
	def hist_eql(self):
		img = np.array(self.current_image)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) 
		img = hist_eql_alg(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA) 
		self.current_image = Image.fromarray(img)  # convert image for PIL
		imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
		self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
		self.panel.config(image=imgtk)  # show the image
		
	# Start/Stop record video
	def record_video(self):
		if self.status == 1 and self.record_video_status == 0:
			print('[INFO] recording')
			self.btn_3['text'] = 'Stop Recording'
			fourcc = cv2.VideoWriter_fourcc(*'mp4v') # decode method
			# Get current width of frame
			ts = datetime.datetime.now() # grab the current timestamp
			filename = '{}.mp4'.format(ts.strftime('%Yy-%mm-%dd_%Hh-%Mm-%Ss'))  # construct filename
			path = os.path.join(self.output_path, filename)  # construct output path
			# Get the width and height of frame
			width = int(self.webcam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
			height = int(self.webcam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
			self.video_out = cv2.VideoWriter(path,fourcc, 20.0,(width,height))
			self.record_video_status = 1 # start record
		elif self.status == 1 and self.record_video_status == 1:
			print('[INFO] stop record')
			self.btn_3['text'] = 'Start Record Video'
			self.record_video_status = 0 # stop record
			self.video_out.release()

	# Open video from file
	def open_video(self):
		self.status = 3
		file_path = filedialog.askopenfilename()
		if file_path != '':
			print('[INFO] Video file path: '+file_path)
			try:
				self.video_buffer = cv2.VideoCapture(file_path)
				while(self.video_buffer.isOpened()):
					ret, frame = self.video_buffer.read()
					cv2.imshow('Press Q to exit',frame)
					time.sleep(0.1)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				self.video_buffer.release()
			except Exception as e:
				self.video_buffer.release()
				self.replay(file_path)

	def replay(self, path):
		try:
			self.video_buffer = cv2.VideoCapture(path)
			while(self.video_buffer.isOpened()):
				ret, frame = self.video_buffer.read()
				cv2.imshow('Press Q to exit',frame)
				time.sleep(0.1)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			self.video_buffer.release()
		except Exception as e:
			self.video_buffer.release()
			self.replay(path)

	# Open image from file
	def open_image(self):
		self.status = 2
		file_path = filedialog.askopenfilename()
		if file_path != '':
			print('[INFO] Image file path: '+file_path)
			try:
				img = cv2.imread(file_path,1)
				img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
				self.current_image = Image.fromarray(img)  # convert image for PIL
				imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
				self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
				self.panel.config(image=imgtk)  # show the image
			except Exception as e:
				print('[INFO] Cannot open!')

	# Open webcam button handler
	def open_webcam(self):
		if self.status != 1:
			self.status = 1
			self.webcam = cv2.VideoCapture(0)
			self.btn_1['text'] = 'Close Webcam'
			self.video_loop()
		else:
			self.btn_1['text'] = 'Open Webcam'
			self.webcam.release()
			self.status = 0

	# Display webcam image in tkinter panel (refresh itself periodically) 
	def video_loop(self):
		if self.status != 1:
			return # Stop the loop
		ok, frame = self.webcam.read()  # read frame from video stream
		if ok:  # frame captured without any errors
			img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
			img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
			self.current_image = Image.fromarray(img)  # convert image for PIL
			imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
			self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
			self.panel.config(image=imgtk)  # show the image
			if self.record_video_status == 1:
				'''record video'''
				print('[INFO] recording')
				self.video_out.write(frame)
		self.root.after(30, self.video_loop)  # call itself (video_loop) after 30ms

	# Take snapshot and save it to file (in the same folder as the source code)
	def take_snapshot(self):
		ts = datetime.datetime.now() # grab the current timestamp
		filename = '{}.png'.format(ts.strftime('%Yy-%mm-%dd_%Hh-%Mm-%Ss'))  # construct filename
		path = os.path.join(self.output_path, filename)  # construct output path
		self.current_image.save(path, 'PNG')  # save image as jpeg file
		print('[INFO] saved {}'.format(filename))

	# destroy the root object, release all resources
	def destructor(self):
		self.root.destroy()
		self.video_buffer.release()
		self.webcam.release()  # release web camera
		if self.video_out != None:
			self.video_out.release()
		cv2.destroyAllWindows()  # it is not mandatory in this application

# start the GUI
gui = GUI()
gui.root.mainloop()