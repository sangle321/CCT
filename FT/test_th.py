import numpy as np
import cv2
from skimage.filters import threshold_yen
from PIL import Image as im, ImageDraw
import matplotlib.pyplot as plt


def clear_color(path):
    path_gamma = '_gamma.jpg'
    path_gray_final = '_gray_final.jpg'
    path_output = '_output_final.jpg'
    path_gray = 'gray.jpg'

    img = cv2.imread(path)
    img_gamma = img.copy()

    hsv = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)
    # cv2.imwrite(path_hsv, hsv)
    gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_gray, gray)
    gray_final = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path_gray_final, gray_final)

    threshold_max = threshold_yen(gray_final)

    if threshold_max != 255:
        threshold_min = threshold_yen(gray)
        threshold_color = (threshold_max + threshold_min) / 2
        print(threshold_min, threshold_max, threshold_color)
        lower_black = np.array([threshold_color], dtype="uint16")
        upper_black = np.array([255], dtype="uint16")
        black_mask = cv2.inRange(gray_final, lower_black, upper_black)
        cv2.imwrite(path_output, black_mask)
    else:
        cv2.imwrite(path_output, gray_final)
    return path_output, img

def CCT(path):
    path_img, img = clear_color(path)
    thresh = cv2.imread(path_img, 0)
    img2 = img.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # 3,3
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.000000000001 * cv2.arcLength(contour, True), True)
        # print(approx)
        cv2.drawContours(img2, [approx], 0, (128, 128, 128), 1)

    cv2.imshow("img", img2)
    cv2.waitKey(0)
    gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.medianBlur(gray_img, 19)

    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.5, 55, param1=200, param2=45, minRadius=10, maxRadius=70)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 255, 255), 3)

    # cv2.imshow("img2", thresh)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    plt.imshow(img)
    plt.show()


CCT("ong4.jpg")

