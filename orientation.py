import cv2
import numpy as np
from scipy.ndimage import rotate

def get_template():
    image = cv2.imread('./data/CC Lake/DSC00839.jpg', 0)
    m, n = image.shape
    print(m, n)

    # new shape = (1224, 918)
    # image1 =
    image1 = cv2.flip(cv2.resize(image, dsize = (612, 459)), 1)
    print(image1.shape)

    # slice template
    template = image1[200:245, 250:425]
    cv2.imwrite("./data/CC Lake/template.png", template)
    cv2.imshow('a', template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# correlate the template
def rotation():
    temp = cv2.imread('./data/CC Lake/template.png', 0)
    w, h = temp.shape
    print(w, h)

    image1 = cv2.imread('./data/CC Lake/DSC00948.jpg', 0)
    image1 = cv2.imread('./data/CC Lake/DSC00958.jpg', 0)
    image1 = cv2.resize(image1, dsize = (612, 459))
    W, H = image1.shape
    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_CCORR
    # for i in range(36):
    image1_flip = cv2.flip(image1, 1)
    for i in range(36):
        # the rotate takes angle in degrees
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

    # temp_w = best_temp.shape[1]
    # temp_h = best_temp.shape[0]
    result_cols = W - w + 1
    result_rows = H - h + 1
    print(result_cols, result_rows)
    cv2.imshow('a', best_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # get_template()
    rotation()
