import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import argparse
import imutils

img = cv2.imread("samples/ong7.jpg")
img2 = img
# shifted = cv2.pyrMeanShiftFiltering(img, 10, 100)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)

ret = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

print(ret)

high_th = ret[0]
low_th = ret[0] * 1/2

canny_img = cv2.Canny(gray, low_th, high_th)

cv2.imshow('circles', canny_img)
cv2.waitKey()




# contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # print(contours)
# for contour in contours:
#     peri = cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, 0.000000000001 * peri, True)
#     # print(approx)
#     (x, y, w, h) = cv2.boundingRect(contour)
#     ar = w / float(h)
#     if len(approx) > 20:
#         cv2.drawContours(img2, [approx], 0, (0, 255, 0), 5)
#
# cv2.imshow("img", morph)
# cv2.waitKey(0)


#
# gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)


# blur_hor = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
# blur_vert = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
# mask = ((img[:,:,0] > blur_hor*1.2) | (img[:, :, 0] > blur_vert*1.2)).astype(np.uint8)*255
#
# mask = cv2.GaussianBlur(mask, (9, 9), cv2.BORDER_CONSTANT)
#
# circles = cv2.HoughCircles(mask,
#                            cv2.HOUGH_GRADIENT,
#                            minDist=55,
#                            dp=1.5,
#                            param1=200,
#                            param2=85,
#                            minRadius=10,
#                            maxRadius=70 )
#
# circles = np.uint16(np.round(circles))
# output = img.copy()
#
# list_radius = [i[2] for i in circles[0, :]]
# for i in circles[0, :]:
# 	center = (i[0], i[1])
# 	popular_r = CountFrequency(list_radius)
# 	general_radius = np.average([popular_r[i] for i in range(0, 2)])
# 	if i[2] > general_radius / 2:
# 		cv2.circle(output, center, int(general_radius / 2), (0, 255, 0), int(general_radius / 3))  # circle center
# 		cv2.circle(output, center, int(general_radius), (0, 255, 255), 3)  # circle outline
#
# cv2.imshow("img", output)
# cv2.waitKey(0)
#
# plt.imshow(output)
# plt.show()
