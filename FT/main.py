import numpy as np
import cv2
from skimage.filters import threshold_yen
import matplotlib.pyplot as plt
import log


class PipeCounter():

    def __init__(self, path, gen_radius):
        """
        :param path: path of image
        :param gen_radius: General radius format for pipecounter
        """
        self.path = path
        self.gen_radius = gen_radius

        self.log_cct = log.create_logger_cct()
        self.request_log = "[CCT_CMD] [REQUEST_USER]"

    def clear_color(self):
        """
        :description: This function is used to get a reasonable threshold for the image
        :return: path_threshold_image, image_origin
        """
        try:
            path_output = '_output_final.jpg'
            img = cv2.imread(self.path)
            img_gamma = img.copy()

            hsv = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)

            gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
            gray_final = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2GRAY)

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
        except Exception as err:
            self.log_cct.info(self.request_log + "] ERROR {clear_color()} < " + str(err) + " >")


    def CCT(self):
        try:
            path_img, img = self.clear_color()
            thresh = cv2.imread(path_img, 0)
            img2 = img.copy()

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))  # 3,3
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.000000000001 * cv2.arcLength(contour, True), True)
                cv2.drawContours(img2, [approx], 0, (128, 128, 128), 1)

            gray_img = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.medianBlur(gray_img, 19)

            circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.5, 55, param1=200, param2=45, minRadius=10, maxRadius=70)

            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # draw the outer circle
                cv2.circle(img, center, self.gen_radius, (0, 255, 0), int(self.gen_radius/3))
                # draw the center of the circle
                cv2.circle(img, center, 2, (0, 255, 255), 3)

            plt.imshow(img)
            plt.show()
        except Exception as err:
            self.log_cct.info(self.request_log + "] ERROR {CCT()} < " + str(err) + " >")

    def main_cct(self):
        try:
            self.CCT()
        except Exception as err:
            self.log_cct.info(self.request_log + "] ERROR {main_cct()} < " + str(err) + " >")

obj = PipeCounter("img3.png", 30)

obj.main_cct()




