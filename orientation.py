import cv2
import numpy as np
from scipy.ndimage import rotate

def get_template():
    # image = cv2.imread('../data/CC Lake/DSC00839.jpg')
    image = cv2.imread('../data/CC Lake/DSC00840.jpg')
    # m, n = image.shape
    # print(m, n)

    # new shape = (1224, 918)
    image1 = cv2.flip(cv2.resize(image, dsize = (1224, 918)), 1)
    # print(image1.shape)

    # slice template, 90*450 or 1by5
    template = image1[380:485, 470:995]
    cv2.imwrite("../data/templates/template_color2.png", template)
    cv2.imwrite("../data/templates/template_color_right2.png", cv2.flip(template, 1))
    cv2.imshow('a', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# tuning the bgr inrange for edge:
def ToBNW(image_path, ksize = 5):
    image = cv2.imread(image_path)
    # image = cv2.resize(image, dsize = (1224, 918))
    # image = cv2.resize(image, None, fx = 0.5, fy = 0.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255), mask o yellow (15,0,0) ~ (36, 255, 255)
    rgbmask = cv2.inRange(image, (80,0,0), (200, 250, 250))
    mask = cv2.inRange(hsv, (0,0,0), (90, 240, 240))

    rgb1 = np.copy(rgbmask).astype(np.int)
    hsv1 = np.copy(mask).astype(np.int)
    hsv2 = np.abs(255 - hsv1)
    # print(rgb1)
    # print(hsv1)
    # print(hsv2)
    # print(mask.dtype)
    combine = np.uint8((hsv2 + rgb1)/2)
    combine[combine < 130] = 0
    # print(combine.dtype)
    mask = cv2.blur(combine, ksize = (ksize, ksize))
    edge = cv2.Canny(mask, 20, 200)
    # lines = cv2.HoughLinesP(edge, 1, np.pi/180, 80, 30, 10 );
    # print(lines)
    # for i in lines:
    #     # print(i)
    #     x1 = i[0][0]
    #     y1 = i[0][1]
    #     x2 = i[0][2]
    #     y2 = i[0][3]
    #     cv2.line(edge,(x1,y1),(x2, y2),(0,0,0),2)
    # cv2.imshow('a', combine)
    # cv2.imshow('c', image)
    # cv2.imshow('b', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge

def contour():
    image = cv2.imread(image_path)
    image = cv2.flip(cv2.resize(image, dsize = (1224, 918)), 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# correlate the template
def rotation():
    temp = ToBNW('../data/templates/template_color1.png', ksize = 3)
    # temp = imread()
    temp = cv2.resize(temp, dsize = None, fx = 0.5, fy = 0.5)
    w, h = temp.shape[0:2]
    print(w, h)
    # cv2.imshow('a', temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image1 = cv2.imread('../data/CC Lake BW/Cond3/DSC00839.jpg')
    # image1 = ToBNW('../data/CC Lake/DSC00839.jpg')
    image1 = cv2.resize(image1, dsize = (1224, 918))
    image1 = cv2.resize(image1, dsize = None, fx = 0.5, fy = 0.5)
    W, H = image1.shape[0:2]
    print(W, H)
    image1_flip = cv2.flip(image1, 0)

    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED give correct using color_right template, exactly same
    # if using the left template and flip,
    method = cv2.TM_CCORR_NORMED

    # scale template as well:

    for s in np.arange(0.8, 1.2, 0.1):
        # print(s)
        temp = cv2.resize(temp, None, fx = s, fy = s)
        for i in range(40):
        #     # the rotate takes angle in degrees
            img_rot = rotate(image1, 5 * i, mode = 'constant', reshape = False, cval = 0)
            img_rotflip = rotate(image1_flip, 5 * i, mode = 'constant', reshape = False, cval = 0)
            # set reshape = true, cval = 255 good for part 4-1,2,4...
            # print(img_rot.dtype, temp.dtype)
            res_rot = cv2.matchTemplate(img_rot, temp, method)
            res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
            # print(res_rot)
            if res_rot.max() >= best_match_score:
                best_match_score = res_rot.max()
                res = res_rot
                best_img = img_rot
                info = ('image_rot', i, s)
            elif res_rotflip.max() >= best_match_score:
                best_match_score = res_rotflip.max()
                res = res_rotflip
                best_img = img_rotflip
                info = ('image_rotflip', i, s)
    print(res, best_match_score, best_img.shape)
    print(info)
    #
    # # temp_w = best_temp.shape[1]
    # # temp_h = best_temp.shape[0]
    # result_cols = W - w + 1
    # result_rows = H - h + 1
    # print(result_cols, result_rows)
    cv2.imshow('a', best_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

from os import listdir
from os.path import isfile, join
from pathlib import Path

def AllBNW():
    path = "../data/CC Lake BW/Cond3"
    Path(path).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir('../data/CC Lake/') if isfile(join('../data/CC Lake/', f))]
    # print(onlyfiles)
    for f in onlyfiles:
        fname = join('../data/CC Lake/', f)
        output = ToBNW(fname)
        cv2.imwrite(join(path, f), output)

if __name__ == '__main__':
    # get_template()
    rotation()
    # ToBNW('../data/CC Lake/DSC00839.jpg')
    # AllBNW()
