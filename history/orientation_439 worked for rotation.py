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
def ToBNW(image_path):
    image = cv2.imread(image_path)
    # image1 = cv2.flip(cv2.resize(image, dsize = (1224, 918)), 1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255), mask o yellow (15,0,0) ~ (36, 255, 255)
    mask = cv2.inRange(hsv, (0,0,0), (100, 255, 255))
    edge = cv2.Canny(mask, 20, 250)

    # cv2.imshow('b', cv2.resize(edge, None, fx = 0.25, fy = 0.25))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edge

# correlate the template
def rotation():
    temp = ToBNW('../data/templates/template_color1.png')
    temp = cv2.resize(temp, dsize = None, fx = 0.5, fy = 0.5)
    w, h = temp.shape[0:2]
    print(w, h)

    image1 = ToBNW('../data/CC Lake/DSC00839.jpg')
    image1 = cv2.resize(image1, dsize = (1224, 918))
    image1 = cv2.resize(image1, dsize = None, fx = 0.5, fy = 0.5)
    W, H = image1.shape[0:2]
    print(W, H)

    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED give correct using color_right template, exactly same
    # if using the left template and flip,
    method = cv2.TM_CCORR_NORMED
    # for i in range(36):
    image1_flip = cv2.flip(image1, 0)
    for i in range(36):
    #     # the rotate takes angle in degrees
        img_rot = rotate(image1, 10 * i, mode = 'constant', reshape = False, cval = 0)
        img_rotflip = rotate(image1_flip, 10 * i, mode = 'constant', reshape = False, cval = 0)
        # set reshape = true, cval = 255 good for part 4-1,2,4...
        res_rot = cv2.matchTemplate(img_rot, temp, method)
        res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
        # print(res_rot)
        if res_rot.max() >= best_match_score:
            best_match_score = res_rot.max()
            res = res_rot
            best_img = img_rot
            info = ('image_rot', i)
        elif res_rotflip.max() >= best_match_score:
            best_match_score = res_rotflip.max()
            res = res_rotflip
            best_img = img_rotflip
            info = ('image_rotflip', i)
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

if __name__ == '__main__':
    # get_template()
    rotation()
    # ToBNW('../data/CC Lake/DSC00839.jpg')
