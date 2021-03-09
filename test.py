import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_yen, threshold_otsu, threshold_li, try_all_threshold

image = cv2.imread("samples/img4.png", 0)

from skimage.filters import try_all_threshold

fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
plt.show()


# def nothing(x):
#   pass
#
# cv2.namedWindow('Colorbars')
# hh = 'Max'
# hl = 'Min'
# wnd = 'Colorbars'
# cv2.createTrackbar("Max", "Colorbars", 0, 255, nothing)
# cv2.createTrackbar("Min", "Colorbars", 0, 255, nothing)
# img = cv2.imread('samples/img4.png', 0)
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#
# while(1):
#    hul = cv2.getTrackbarPos("Max", "Colorbars")
#    huh = cv2.getTrackbarPos("Min", "Colorbars")
#    ret, thresh1 = cv2.threshold(img, hul, huh,  cv2.THRESH_OTSU)
#    print(ret)
#    ret, thresh2 = cv2.threshold(img, hul, huh, cv2.THRESH_BINARY_INV)
#    ret, thresh3 = cv2.threshold(img, hul, huh, cv2.THRESH_TRUNC)
#    ret, thresh4 = cv2.threshold(img, hul, huh, cv2.THRESH_TOZERO)
#    ret, thresh5 = cv2.threshold(img, hul, huh, cv2.THRESH_TOZERO_INV)
#    # cv2.imshow(wnd)
#    cv2.imshow("thresh1", thresh1)
#    cv2.imshow("thresh2", thresh2)
#    cv2.imshow("thresh3", thresh3)
#    cv2.imshow("thresh4", thresh4)
#    cv2.imshow("thresh5", thresh5)
#
#    k = cv2.waitKey(1) & 0xFF
#    if k == ord('m'):
#      mode = not mode
#    elif k == 27:
#      break
# cv2.destroyAllWindows()