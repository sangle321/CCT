import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as im, ImageDraw
import argparse
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="path to input derectory of images")
args = vars(ap.parse_args())

def CountFrequency(my_list):
	# Creating an empty dictionary
	count = {}
	for i in my_list:
		count[i] = count.get(i, 0) + 1
	my_dict = {k: v for k, v in sorted(count.items(), key=lambda item: item[1], reverse=True)}
	return list(my_dict.keys())

def get_color(path):
    img = im.open(path)
    colors = img.getcolors(256) #put a higher value if there are many colors in your image
    max_occurence, most_present = 0, 0
    try:
        for c in colors:
            if c[0] > max_occurence:
                (max_occurence, most_present) = c
        return most_present
    except TypeError:
        return 'color'
    
def resize_image(path):
	width_g, height_g = 2048, 1536
	img = cv2.imread(path)
	in_w = img.shape[0]
	in_h = img.shape[1]
	if in_w < width_g/2 and in_h < height_g/2:
		raise ValueError("Input image fail !")
	elif in_w > width_g and in_h > height_g:
		img = cv2.resize(img, dsize=(width_g, height_g), interpolation=cv2.INTER_AREA)
	return img

def write_file_xy(storage):
	f = open("DEV/output.txt", "w")
	for c in storage[0, :]:
		f.write(str(c[0]) + "\t" + str(c[1]) + "\n")
	f.close()

def get_loc_oxy(storage):
	circles = np.uint16(np.round(storage))
	write_file_xy(circles)
	list_point = []
	list_radius = []
	for c in circles[0, :]:
		list_point.append((c[0], c[1]))
		list_radius.append(c[2])
	popular_r = CountFrequency(list_radius)
	general_radius = np.average([popular_r[i] for i in range(0, 2)])
	return list_point, general_radius

def draw_circles(storage, image):
	cct = 0
	if storage is not None:
		circles = np.uint16(np.round(storage))
		_, general_radius = get_loc_oxy(storage)
		for i in circles[0, :]:
			center = (i[0], i[1])
			radius = int(general_radius) - 1
			if i[2] > radius / 2:
				cct += 1
				cv2.circle(image, center, int(radius / 2), (0, 255, 0), int(radius / 3))  # circle center
				cv2.circle(image, center, radius, (0, 255, 255), 3)  # circle outline
	return image, cct

def CCT(path):
	# img = resize_image(path)
	img = cv2.imread(path)
	img2 = img.copy()
	
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	# h, s, v = cv2.split(hsv_img)
	cv2.imshow("img", hsv_img)
	cv2.waitKey(0)
	gray_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
	# canny_img = cv2.Canny(hsv_img, 130, 260)
	# ret, threshold = cv2.threshold(canny_img, 0, 255, cv2.THRESH_OTSU)
	threshold = threshold_yen(gray_img)
	bright = rescale_intensity(gray_img, (0, threshold), (0, 255))
	cv2.imshow("img", bright)
	cv2.waitKey(0)
	# print(ret)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 3,3
	morph = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

	contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# print(contours)
	for contour in contours:
		approx = cv2.approxPolyDP(contour, 0.0000000000000000000001 * cv2.arcLength(contour, True), True)
		# print(approx)
		cv2.drawContours(img2, [approx], 0, (128, 128, 128), 2)
	
	cv2.imshow("img", img2)
	cv2.waitKey(0)
	gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	gray_img = cv2.medianBlur(gray_img, 5)
	cv2.imshow("img", gray_img)
	cv2.waitKey(0)
	circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.5, 35, param1=200, param2=38, minRadius=10, maxRadius=50)
	return draw_circles(circles, img)

def main_CCT():
	path = args["image"]
	try:
		img, cct = CCT(path)
		plt.imshow(img)
		plt.show()
	except Exception as err:
		print(repr(err))
	
main_CCT()