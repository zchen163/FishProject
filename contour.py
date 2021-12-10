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
    onlyfiles = [f for f in listdir('../data/CC Lake extracted/2/') if isfile(join('../data/CC Lake extracted/2/', f))]
    # samples = ['DSC00899.jpg']
    path = "../data/CC Lake extracted/2/"
    for f in onlyfiles[:5]:
        image = cv2.imread(join(path, f))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # key is to find a binary template before findContours!
        # one option is hsv sat (0, 70)
        sat = hsv[:, :, 1]
        sat = cv2.medianBlur(sat, ksize = 3)
        sat1 = cv2.inRange(sat, (0), (60))


        rmask = cv2.inRange(image[:, :, 0], (100), (255))
        # rgbmask = cv2.inRange(image, (0,0,0), (0, 0, 0))
        # rgbmask = cv2.medianBlur(rgbmask, ksize = 3)
        # hsv0 = cv2.inRange(hsv, (0,0,50), (100, 250, 250))

        # try canny edge:
        # edge = cv2.Canny(sat1, 100, 200)
        # edge1 = cv2.Canny(rmask, 100, 200)
        # contours, hierarchy  = cv2.findContours(sat1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy  = cv2.findContours(sat1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours1, hierarchy1  = cv2.findContours(rmask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
        # contours1, hierarchy1  = cv2.findContours(rmask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_NONE)
        # img1 = np.copy(image)
        img1 = cv2.drawContours(np.copy(image), contours, -1, (0,255,75), 2)
        img2 = cv2.drawContours(np.copy(image), contours1, -1, (0,255,75), 2)
        # show_image(res_img)

        # count number of points
        # print(type(contours))
        # print(len(contours), type(contours[2]))
        # print(contours[8].shape)
        # print(contours[8])
        '''use skimage contour? '''

        f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        ax1.imshow(sat1)
        ax2.imshow(rmask)
        ax3.imshow(img1)
        ax4.imshow(img2)
        # h = cv2.inRange(hsv, (100, 70, 120), (120, 120, 200))
        # plt.figure(1)
        # plt.imshow(h, cmap=plt.cm.gray)
        plt.show()


        # cv2.imshow('image', img)
        # cv2.imshow('rgb', rgbmask)
        # cv2.imshow('hsv', hsv0)
        # # fname = join("../data/contour/", f)
        # # cv2.imwrite(fname, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # return edge
if __name__ == '__main__':
    extracted()
