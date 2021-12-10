import cv2
import numpy as np
from scipy.ndimage import rotate
from os import listdir
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt


'''adjust the extracted image'''

def extracted():
    # load an extracted and play with orientation again
    samples = ['DSC00899.jpg']
    path = "../data/CC Lake extracted/2/"
    image = cv2.imread(join(path, samples[0]))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # try grayscale first
    # rgbmask = cv2.inRange(image, (20,50,50), (180, 180, 180))
    # rgbmask = cv2.medianBlur(rgbmask, ksize = 3)
    # hsv0 = cv2.inRange(hsv, (0,0,50), (100, 250, 250))
    #
    # rgb1 = np.copy(rgbmask).astype(np.uint8)
    # hsv1 = np.copy(hsv0).astype(np.int)
    # hsv2 = np.abs(255 - hsv1)
    # # print(rgb1)
    # # print(hsv1)
    # # print(hsv2)
    # # print(mask.dtype)
    # combine = np.uint8((hsv2 + rgb1)/2)
    # combine[combine < 130] = 0
    # # print(combine.dtype)
    # mask = cv2.medianBlur(combine, ksize = 5)
    # edge = cv2.Canny(mask, 100, 200)
    # kernel = np.ones((5,5), np.uint8)
    # dilate = cv2.dilate(edge,kernel,iterations = 1)
    #     image = cv2.imread(join(path, i))
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     gray = cv2.imread(join(path, i), 0)
    #     # hsvmask = cv2.inRange(hsv, (0,0,10), (50, 250, 250))
    #
    #     colormask = cv2.inRange(image, (50,50,50), (200, 200, 150))
    #     # find the eye circle!
    #     # first get edges
    #     # lower_yellow = np.array([0,100,100])
    #     # upper_yellow = np.array([20,255,255])
    #     # graymask = cv2.inRange(gray, 150, 255)
    #     # edge = cv2.Canny(hsvmask, 100, 200)
    #     #dp = 1, minDist = 20, param1 = 100 or 50, param2 = 22,  minRadius = 10, maxRadius = 40
    #     # center_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp = 1, minDist = radii_range[0]*2, param1 = 50, param2 = 20,  minRadius = radii_range[0], maxRadius = radii_range[-1]+6)
    #
    #
    # cv2.imshow('image1', image[:, :, 0])
    # cv2.imshow('image2', image[:, :, 1])
    sat = hsv[:, :, 1]
    sat1 = cv2.inRange(sat, (0), (50))
    # sat1 = cv2.medianBlur(sat1, ksize = 3)
    # sat between 0 and 50 is pretty good...

    # contours, hierarchy  = cv2.findContours(sat1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy  = cv2.findContours(sat1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img1 = np.copy(image)
    img = cv2.drawContours(img1, contours, -1, (0,255,75), 2)
    # show_image(res_img)
    print(type(contours))
    # print(len(contours), type(contours[2]))
    print(contours[8].shape)
    print(contours[8])


    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(img, cmap=plt.cm.gray)
    # # ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    # # ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    # ax.set_xticks([]), ax.set_yticks([])
    # ax.axis([0, sat1.shape[1], sat1.shape[0], 0])
    #
    # plt.show()
    # cv2.imshow('image3', sat1)
    # #     # cv2.imshow('gs', gray)
    # #     # cv2.imshow('colormask', colormask)
    # #     # cv2.imshow('gsmask', graymask)
    # #     # print(hsv.shape)
    # # cv2.imshow('blur', dilate)
    cv2.imshow('edge', img)
    cv2.imshow('image', image)
    cv2.imwrite("../data/contour/DSC00899.jpg", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return edge
if __name__ == '__main__':
    extracted()
