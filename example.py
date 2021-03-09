import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


img = cv2.imread("samples/img4.png")
cimg = img.copy()
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
h, s, v = cv2.split(gray_img)

blur = cv2.GaussianBlur(v, (3, 3), 0)

ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU)
print(ret)
sobel = cv2.Sobel(th, cv2.CV_8UC1, 1, 1, ksize=5)


cv2.imwrite('sobel.jpg', sobel)
gray = cv2.imread('sobel.jpg', 0)

# cimg = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)


circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.4,30,
                            param1=200,param2=30,minRadius=8,maxRadius=50)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()


# _, threshold = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# print(_)
# circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1.5, 20,
#                           param1=200, param2=40,
#                           minRadius=5, maxRadius=60)
#
# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         center = (i[0], i[1])
#         # circle center
#         cv2.circle(img2, center, 1, (0, 100, 100), 3)
#         # circle outline
#         radius = i[2]
#         cv2.circle(img2, center, radius, (255, 0, 255), 3)
#
# cv2.imshow("detected circles", img2)
# cv2.waitKey(0)
