import cv2
import numpy as np
from scipy.ndimage import rotate

def get_template():
    image = cv2.imread('../data/CC Lake/DSC00839.jpg')
    # image = cv2.imread('../data/CC Lake/DSC00840.jpg')
    # m, n = image.shape
    # print(m, n)

    # new shape = (1224, 918)
    image1 = cv2.flip(cv2.resize(image, dsize = (1224, 918)), 1)
    # print(image1.shape)

    # slice template, 90*450 or 1by5
    template = image1[380:485, 470:995]
    cv2.imwrite("../data/templates/template_color1.png", template)
    cv2.imwrite("../data/templates/template_color_right1.png", cv2.flip(template, 1))
    cv2.imshow('a', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# tuning the bgr inrange for edge:
def ToBNW(image_path, fx = 1, ksize = 5):
    image = cv2.imread(image_path)
    # image = cv2.resize(image, dsize = (1224, 918))
    if fx != 1:
        image = cv2.resize(image, None, fx = fx, fy = fx)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255), mask o yellow (15,0,0) ~ (36, 255, 255)
    rgbmask = cv2.inRange(image, (80,50,50), (200, 200, 200))
    rgbmask = cv2.medianBlur(rgbmask, ksize = 3)
    hsv0 = cv2.inRange(hsv, (0,0,50), (100, 250, 250))

    rgb1 = np.copy(rgbmask).astype(np.uint8)
    hsv1 = np.copy(hsv0).astype(np.int)
    hsv2 = np.abs(255 - hsv1)
    # print(rgb1)
    # print(hsv1)
    # print(hsv2)
    # print(mask.dtype)
    combine = np.uint8((hsv2 + rgb1)/2)
    combine[combine < 130] = 0
    # print(combine.dtype)
    mask = cv2.medianBlur(combine, ksize = ksize)
    edge = cv2.Canny(mask, 100, 200)
    # lines = cv2.HoughLinesP(edge, 1, np.pi/180, 80, 30, 10 );
    # print(lines)
    # for i in lines:
    #     # print(i)
    #     x1 = i[0][0]
    #     y1 = i[0][1]
    #     x2 = i[0][2]
    #     y2 = i[0][3]
    #     cv2.line(edge,(x1,y1),(x2, y2),(0,0,0),2)
    # cv2.imshow('rgb', hsv)
    # cv2.imshow('ra', image)
    # cv2.imshow('rgb', rgbmask)
    # cv2.imshow('hsv', np.uint8(hsv2))
    # cv2.imshow('combine', combine)
    # cv2.imshow('edge', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return edge
    return combine

# correlate the template
def slice():
    # slice image and remove the ruler thing:
    path = "../data/CC Lake slice/"
    Path(path).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir('../data/CC Lake/') if isfile(join('../data/CC Lake/', f))]
    # print(onlyfiles)
    for f in onlyfiles:
        fname = join('../data/CC Lake/', f)
        image = cv2.resize(cv2.imread(fname), None, fx = 0.25, fy = 0.25)
        print(image.shape) # (918, 1224, 3)
        sliced = image[300:600, 200:1000, :]
        cv2.imwrite(join(path, f), sliced)

def rotation(fname):
    temp = ToBNW('../data/templates/template_color1.png', ksize = 3)
    # temp = imread()
    # temp = cv2.resize(temp, dsize = None, fx = 0.5, fy = 0.5)
    w, h = temp.shape[0:2]
    print(w, h)
    # cv2.imshow('a', temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # image1 = cv2.imread('../data/CC Lake BW/Cond3/DSC00839.jpg')
    image1 = ToBNW(fname, fx = 0.55, ksize = 5)
    # image1 = cv2.resize(image1, dsize = (1224, 918))
    # image1 = cv2.resize(image1, dsize = None, fx = 0.25, fy = 0.25)
    W, H = image1.shape[0:2]
    print(W, H)
    image1_flip = cv2.flip(image1, 1)
    # cv2.imshow('normal', image1)
    # cv2.imshow('flip', image1_flip)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED give correct using color_right template, exactly same
    # if using the left template and flip,
    method = cv2.TM_CCORR_NORMED

    # scale template as well:

    for s in np.arange(0.7, 1.3, 0.05):
        # print(s)
        temp = cv2.resize(temp, None, fx = s, fy = s)
        for i in range(36):
        #     # the rotate takes angle in degrees
            img_rot = rotate(image1, 10 * i, mode = 'constant', reshape = False, cval = 0)
            img_rotflip = rotate(image1_flip, 10 * i, mode = 'constant', reshape = False, cval = 0)
            # set reshape = true, cval = 255 good for part 4-1,2,4...
            # print(img_rot.dtype, temp.dtype)
            res_rot = cv2.matchTemplate(img_rot, temp, method)
            res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
            # print(res_rot)
            if res_rot.max() >= best_match_score:
                best_match_score = res_rot.max()
                res = res_rot
                best_img = img_rot
                info = ('rot', i, s)
            elif res_rotflip.max() >= best_match_score:
                best_match_score = res_rotflip.max()
                res = res_rotflip
                best_img = img_rotflip
                info = ('rotflip', i, s)
    print(best_match_score, best_img.shape)
    print(info)

    # plot best image orientation
    best = cv2.imread(fname)
    best = rotate(best, 10 * info[1], mode = 'constant', reshape = False, cval = 0)
    if info[0] == 'rotflip':
        best = cv2.flip(best, 1)

    cv2.imshow('best', best)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # # temp_w = best_temp.shape[1]
    # # temp_h = best_temp.shape[0]
    # result_cols = W - w + 1
    # result_rows = H - h + 1
    # print(result_cols, result_rows)
    # cv2.imshow('a', best_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

from os import listdir
from os.path import isfile, join
from pathlib import Path

def AllBNW():
    path = "../data/CC Lake BW/Cond1"
    Path(path).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir('../data/CC Lake slice/') if isfile(join('../data/CC Lake slice/', f))]
    # print(onlyfiles)
    for f in onlyfiles[5:15]:
        fname = join('../data/CC Lake slice/', f)
        print(fname)
        output = ToBNW(fname, fx = 0.5, ksize = 3)
        # cv2.imwrite(join(path, f), output)

def test():
    path = "../data/CC Lake BW/Cond1"
    Path(path).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir('../data/CC Lake slice/') if isfile(join('../data/CC Lake slice/', f))]
    # print(onlyfiles)
    for f in onlyfiles[5:15]:
        fname = join('../data/CC Lake slice/', f)
        print(fname)
        output = rotation(fname)

if __name__ == '__main__':
    # get_template()
    # rotation('../data/CC Lake slice/DSC00840.jpg')
    # ToBNW('../data/CC Lake slice/DSC00844.jpg', fx = 0.5)
    # AllBNW()
    # slice()
    test()
