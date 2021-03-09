import cv2
import numpy as np
import matplotlib.pyplot as plt

import log

class COUNTTHING():
	def __init__(self, path):
		self.path = path
		
		self.log_cct = log.create_logger_cct()
		self.request_log = "[CCT_CMD] [REQUEST_USER]"
		self.my_list = None
		self.image = None
		self.circles = None
		self.list_point = []
		self.list_radius = []
		self.cct = None
		self.general_radius = None
		
	def CountFrequency(self):
		try:
			# Creating an empty dictionary
			count = {}
			for i in self.list_radius:
				count[i] = count.get(i, 0) + 1
			my_dict = {k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}
			return list(my_dict.keys())
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {CountFrequency()} < " + str(err) + " >")
	
	def resize_image(self):
		try:
			width_g, height_g = 2048, 1536
			img = cv2.imread(self.path)
			in_w = img.shape[0]
			in_h = img.shape[1]
			if in_w < width_g / 2 and in_h < height_g / 2:
				raise ValueError("Input image fail !")
			elif in_w > width_g and in_h > height_g:
				img = cv2.resize(img, dsize=(width_g, height_g), interpolation=cv2.INTER_AREA)
			return img
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {resize_image()} < " + str(err) + " >")
			
	def write_file_xy(self):
		try:
			f = open("output.txt", "w")
			for c in self.circles[0, :]:
				f.write(str(c[0]) + "\t" + str(c[1]) + "\n")
			f.close()
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {write_file_xy()} < " + str(err) + " >")
	
	def get_loc_oxy(self):
		try:
			self.circles = np.uint16(np.round(self.circles))
			self.write_file_xy()
			for c in self.circles[0, :]:
				self.list_point.append((c[0], c[1]))
				self.list_radius.append(c[2])
			popular_r = self.CountFrequency()
			self.general_radius = np.average([popular_r[i] for i in range(0, 2)])
			return self.list_point, self.general_radius
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {get_loc_oxy()} < " + str(err) + " >")
	
	def draw_circles(self):
		try:
			self.cct = 0
			if self.circles is not None:
				circles = np.uint16(np.round(self.circles))
				_, general_radius = self.get_loc_oxy()
				for i in circles[0, :]:
					center = (i[0], i[1])
					radius = int(general_radius) - 1
					if i[2] > radius / 2:
						self.cct = self.cct + 1
						cv2.circle(self.image, center, int(radius / 2), (0, 255, 0), int(radius / 3))  # circle center
						cv2.circle(self.image, center, radius, (0, 255, 255), 3)  # circle outline
						
			return self.image, self.cct
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {draw_circles()} < " + str(err) + " >")
			
	def CCT(self):
		try:
			self.image = self.resize_image()
			# self.image = cv2.imread(self.path)
			img2 = self.image
			hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
			canny_img = cv2.Canny(hsv_img, 130, 260)
			_, threshold = cv2.threshold(canny_img, 246, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
			
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3,3
			morph = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
			
			contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# print(contours)
			for contour in contours:
				approx = cv2.approxPolyDP(contour, 0.000001 * cv2.arcLength(contour, True), True)
				# print(approx)
				cv2.drawContours(img2, [approx], 0, (128, 128, 128), 1)
			gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
			gray_img = cv2.medianBlur(gray_img, 19)
			self.circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.5, 55, param1=200, param2=39, minRadius=10, maxRadius=70)
			return self.draw_circles()
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {draw_circles()} < " + str(err) + " >")
		
	def main_CCT(self):
		try:
			img, cct = self.CCT()
			plt.imshow(img)
			plt.show()
			return
		except Exception as err:
			self.log_cct.info(self.request_log + "] ERROR {main_CCT()} < " + str(err) + " >")
		
CCT = COUNTTHING("../samples/ong4.jpg")

CCT.main_CCT()