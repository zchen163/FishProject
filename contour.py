import cv2
import numpy as np
from scipy.ndimage import rotate
from os import listdir
from os.path import isfile, join
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.filters import gaussian
from skimage.segmentation import active_contour

'''adjust the extracted image'''

def extracted(fname):
    # load an extracted and play with orientation again
    f = fname
    # samples = ['DSC00899.jpg']
    path = "../data/CC Lake extracted/2/"
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
        # contours1, hierarchy1  = cv2.findContours(rmask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
    contours1, hierarchy1  = cv2.findContours(rmask, cv2.RETR_CCOMP , cv2.CHAIN_APPROX_NONE)
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
    # try 1000 points
    r, c = image.shape[:2]
    initx = np.linspace(0, c, num = 400+1, endpoint = True)
    inity = np.linspace(0, r, num = 100+1, endpoint = True)

    init_top = np.array([np.zeros(400), initx[:400]]).T
    init_right = np.array([inity[:100], np.ones(100)*(c-1)]).T
    init_bot = np.array([np.ones(400)*(r-1), initx[1:]]).T
    init_left = np.array([inity[1:], np.zeros(100)]).T
    init = np.vstack((init_top, init_right, init_bot, init_left))

        # use these 1000 initial points to run active contour:
    # snake1 = active_contour(gaussian(sat1, 3, preserve_range=False),
    #                    init, alpha=0.001, beta=0.01, gamma=0.01)
    # snake2 = active_contour(gaussian(rmask, 3, preserve_range=False),
    #                    init, alpha=0.001, beta=0.01, gamma=0.01)









    f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.imshow(sat1, cmap=plt.cm.gray)
    # ax2.plot(snake1[:, 1], snake1[:, 0], '.b', lw=1)
    ax3.imshow(rmask, cmap=plt.cm.gray)
    # ax4.plot(snake2[:, 1], snake2[:, 0], '.b', lw=1)
        # ax4.axis([0, image.shape[1], image.shape[0], 0])
        # ax1.axis([0, image.shape[1], image.shape[0], 0])
        # ax3.axis([0, image.shape[1], image.shape[0], 0])
        # ax4.imshow(out)
        # h = cv2.inRange(hsv, (100, 70, 120), (120, 120, 200))
        # plt.figure(1)
        # plt.imshow(h, cmap=plt.cm.gray)
    plt.show()


        # cv2.imshow('image', img)
        # cv2.imshow('rgb', rgbmask)
        # cv2.imshow('hsv', hsv0)
        # fname = join("../data/contour/", f)
        # cv2.imwrite(fname, img1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # return edge
def test():
    onlyfiles = [f for f in listdir('../data/CC Lake extracted/2/') if isfile(join('../data/CC Lake extracted/2/', f))]
    for f in onlyfiles[:5]:
        extracted(f)

if __name__ == '__main__':
    # extracted()
    test()
