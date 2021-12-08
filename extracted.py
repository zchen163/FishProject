import cv2
import numpy as np
from scipy.ndimage import rotate
from os import listdir
from os.path import isfile, join
from pathlib import Path



'''adjust the extracted image'''

def extracted():
    # load an extracted and play with orientation again
    samples = ['DSC00839.jpg', 'DSC00857.jpg', 'DSC00861.jpg', 'DSC00866.jpg', 'DSC00871.jpg']
    path = "../data/CC Lake extracted/2/"
    for i in samples[:3]:
        image = cv2.imread(join(path, i))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.imread(join(path, i), 0)
        # hsvmask = cv2.inRange(hsv, (0,0,10), (50, 250, 250))

        colormask = cv2.inRange(image, (50,50,50), (200, 200, 150))
        # find the eye circle!
        # first get edges
        # lower_yellow = np.array([0,100,100])
        # upper_yellow = np.array([20,255,255])
        # graymask = cv2.inRange(gray, 150, 255)
        # edge = cv2.Canny(hsvmask, 100, 200)
        #dp = 1, minDist = 20, param1 = 100 or 50, param2 = 22,  minRadius = 10, maxRadius = 40
        # center_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp = 1, minDist = radii_range[0]*2, param1 = 50, param2 = 20,  minRadius = radii_range[0], maxRadius = radii_range[-1]+6)


        # cv2.imshow('image1', image[:, :, 0])
        # cv2.imshow('image2', image[:, :, 1])
        # cv2.imshow('image3', image[:, :, 2])
        # cv2.imshow('gs', gray)
        # cv2.imshow('colormask', colormask)
        # cv2.imshow('gsmask', graymask)
        # print(hsv.shape)
        cv2.imshow('hsv1', hsv[:, :, 0])
        cv2.imshow('hsv2', hsv[:, :, 1])
        cv2.imshow('hsv3', hsv[:, :, 2])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    extracted()
