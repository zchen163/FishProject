import cv2
import numpy as np
from scipy.ndimage import rotate

def get_template():
    image = cv2.imread('../data/CC Lake/DSC00957.jpg')
    # image = cv2.imread('../data/CC Lake/DSC00840.jpg')
    # m, n = image.shape
    # print(m, n)

    # new shape = (1224, 918)
    image1 = cv2.resize(np.copy(image), dsize = (1224, 918))
    # print(image1.shape)

    # slice template, 90*450 or 1by5
    # image DSC00839.jpg, [390:500, 270:740], 1.png
    # image DSC00866.jpg, [410:540, 450:950], 2.png
    # image DSC00857.jpg, [395:550, 365:1040], 3.png
    # image DSC00957.jpg, [380:530, 410:950], 4.png
    template = image1[380:530, 410:950]
    cv2.imwrite("../data/templates/4.png", cv2.flip(template, 0))
    # cv2.imwrite("../data/templates/template_color_right1.png", cv2.flip(template, 1))
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

def match(image, temp0, slow=0.7, shigh=1.3, method = cv2.TM_CCORR_NORMED):
    # get flipped image
    best_match_score = 0
    image1 = np.copy(image)
    image1_flip = cv2.flip(image1, 1)
    # change template sizes
    for s in np.arange(slow, shigh, 0.05):
        temp = cv2.resize(temp0, None, fx = s, fy = s)
        # search for angles
        for i in [-3, -2, -1, 0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21]:
        # for i in range(0, 36):
        #     # the rotate takes angle in degrees
            img_rot = rotate(image1, 10 * i, mode = 'constant', reshape = False, cval = 0)
            img_rotflip = rotate(image1_flip, 10 * i, mode = 'constant', reshape = False, cval = 0)
            res_rot = cv2.matchTemplate(img_rot, temp, method)
            res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
            if res_rot.max() >= best_match_score:
                best_match_score = res_rot.max()
                res = res_rot
                # best_img = img_rot
                info = ('rot', i, s, temp)
            elif res_rotflip.max() >= best_match_score:
                best_match_score = res_rotflip.max()
                res = res_rotflip
                # best_img = img_rotflip
                info = ('rotflip', i, s, temp)
    return best_match_score, res, info

def match1(image, temp0, slow=0.7, shigh=1.3, method = cv2.TM_CCORR_NORMED):
    # get flipped image
    best_match_score = 0
    image1 = np.copy(image)
    image1_flip = cv2.flip(image1, 1)
    # change template sizes
    for s in np.arange(slow, shigh, 0.05):
        temp = cv2.resize(temp0, None, fx = s, fy = s)
        # search for angles
        for i in [-3, -2, -1, 0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 21]:
        # for i in range(0, 36):
        #     # the rotate takes angle in degrees
            img_rot = rotate(image1, 10 * i, mode = 'constant', reshape = False, cval = 0)
            img_rotflip = rotate(image1_flip, 10 * i, mode = 'constant', reshape = False, cval = 0)
            res_rot = cv2.matchTemplate(img_rot, temp, method)
            res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
            if res_rot.max() >= best_match_score:
                best_match_score = res_rot.max()
                res = res_rot
                best_img = img_rot
                info = ('rot', i, s, temp)
            elif res_rotflip.max() >= best_match_score:
                best_match_score = res_rotflip.max()
                res = res_rotflip
                best_img = img_rotflip
                info = ('rotflip', i, s, temp, best_img)
    return best_match_score, res, info

def rotation(fname):
    temp1 = ToBNW('../data/templates/1.png', fx = 1, ksize = 3)
    temp2 = ToBNW('../data/templates/2.png', fx = 1, ksize = 3)
    temp3 = ToBNW('../data/templates/3.png', fx = 1, ksize = 3)
    temp4 = ToBNW('../data/templates/4.png', fx = 1, ksize = 3)
    # temp = cv2.imread('../data/templates/template_color1.png')
    # h, w = temp0.shape[0:2]
    templst = [temp1, temp2, temp3, temp4]
    image = ToBNW(fname, fx = 1, ksize = 5)

    H, W = image.shape[0:2]
    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_CCORR_NORMED

    # scale template as well
    # star with temp1
    scorelst, reslst, infolst = [], [], []

    score1, res1, info1 = match(image, temp1, slow=0.7, shigh=1.3)
    scorelst.append(score1)
    reslst.append(res1)
    infolst.append(info1)
    score2, res2, info2 = match(image, temp2, slow=0.7, shigh=1.3)
    scorelst.append(score2)
    reslst.append(res2)
    infolst.append(info2)
    score3, res3, info3 = match(image, temp3, slow=0.7, shigh=1.2)
    scorelst.append(score3)
    reslst.append(res3)
    infolst.append(info3)
    score4, res4, info4 = match(image, temp3, slow=0.7, shigh=1.2)
    scorelst.append(score4)
    reslst.append(res4)
    infolst.append(info4)
    # print('temp1', best_match_score)
    # print(score1, info1[:3])
    # print(score2, info2[:3])
    # print(score3, info3[:3])

    idx = np.array(scorelst).argmax()
    print(scorelst, idx)
    manual = None
    if scorelst[idx] < 0.91:
        print('This Sample need manual processing! ')
        print('---------')
        manual = fname
    h, w = infolst[idx][3].shape[0:2]
    # print(infolst[idx][:3])
    #
    #
    # plot best image orientation
    best = cv2.imread(fname)
    if infolst[idx][0] == 'rotflip':
        best = cv2.flip(best, 1)
    best = rotate(best, 10 * infolst[idx][1], mode = 'constant', reshape = False, cval = 0)
    #
    #
    loc = np.unravel_index(reslst[idx].argmax(), reslst[idx].shape)
    # print(loc)
    # print(info[3].shape, 'temp w and h', w, h, 'image W and H', W, H)
    # # # print(info)
    # get the corresponding window
    cornerr = loc[0]
    cornerc = loc[1]
    # image_out = cv2.circle(best, (cornerc, cornerr), radius=5, color=(0, 0, 255), thickness=1)
    # image_out = cv2.circle(image_out, (cornerc + w, cornerr + h), radius=5, color=(0, 0, 255), thickness=-1)

    window = best[cornerr:cornerr + h, cornerc: cornerc + w, :]

    # cv2.imshow('best', best)
    # cv2.imshow('best img', best_img)
    # cv2.imshow('image_out', image_out)
    # # cv2.imshow('a', window)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return best
    return best, window, manual

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
    path_oriented = "../data/CC Lake oriented/6/"
    path_extracted = "../data/CC Lake extracted/3/"
    Path(path_oriented).mkdir(parents=True, exist_ok=True)
    Path(path_extracted).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir('../data/CC Lake slice/') if isfile(join('../data/CC Lake slice/', f))]
    # print(onlyfiles)
    examples = ['DSC00839.jpg', 'DSC00857.jpg', 'DSC00861.jpg', 'DSC00866.jpg', 'DSC00871.jpg']
    # for f in examples:
    manuallst = []
    for f in onlyfiles[30:]:
        fname = join('../data/CC Lake slice/', f)
        print('----Now working on ----')
        print(f)
        best, window, manual = rotation(fname)
        cv2.imwrite(join(path_oriented, f), best)
        cv2.imwrite(join(path_extracted, f), window)
        if manual is not None:
            manuallst.append(manual)
    print('Please process the following samples manually: ')
    for j in manuallst:
        print(j)

def rotate48(fname):
    temp0 = ToBNW('../data/templates/template_color1.png', fx = 1, ksize = 3)
    h, w = temp0.shape[0:2]

    image1 = ToBNW(fname, fx = 1, ksize = 5)
    H, W = image1.shape[0:2]
    # print(W, H)
    image1_flip = cv2.flip(image1, 1)
    # start template matching, and look for different angles:
    best_match_score = 0
    res = None
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED give correct using color_right template, exactly same
    # if using the left template and flip,
    method = cv2.TM_CCORR_NORMED

    # scale template as well:
    scorelst = []
    flipscorelst = []
    for s in [1.2]:
        # print(s)
        # temp = temp0
        temp = cv2.resize(temp0, None, fx = s, fy = s)
        for i in range(0, 36):
        #     # the rotate takes angle in degrees
            img_rot = rotate(image1, 10 * i, mode = 'constant', reshape = False, cval = 0)
            img_rotflip = rotate(image1_flip, 10 * i, mode = 'constant', reshape = False, cval = 0)
            # set reshape = true, cval = 255 good for part 4-1,2,4...
            # print(img_rot.dtype, temp.dtype)
            res_rot = cv2.matchTemplate(img_rot, temp, method)
            scorelst.append((res_rot.max(), i))
            res_rotflip = cv2.matchTemplate(img_rotflip, temp, method)
            flipscorelst.append((res_rotflip.max(), i))
            # print(res_rot)
            if res_rot.max() >= best_match_score:
                best_match_score = res_rot.max()
                res = res_rot
                best_img = img_rot
                info = ('rot', i, s, temp)
            elif res_rotflip.max() >= best_match_score:
                best_match_score = res_rotflip.max()
                res = res_rotflip
                best_img = img_rotflip
                info = ('rotflip', i, s, temp)
    print(best_match_score, best_img.shape)
    print(info[:3])

    print('------------')
    for i in flipscorelst:
        print(i)
    # print(flipscorelst)
    # plot best image orientation
    # best = cv2.imread(fname)
    # best = rotate(best, 10 * info[1], mode = 'constant', reshape = False, cval = 0)
    # if info[0] == 'rotflip':
    #     best = cv2.flip(best, 1)
    # print('shapes of template and image', info[3].shape, best.shape)
    # cv2.imshow('best', best)
def display(fname):

    temp1 = ToBNW('../data/templates/1.png', fx = 1, ksize = 3)
    temp2 = ToBNW('../data/templates/2.png', fx = 1, ksize = 3)
    temp3 = ToBNW('../data/templates/3.png', fx = 1, ksize = 3)
    temp4 = ToBNW('../data/templates/4.png', fx = 1, ksize = 3)
    templst = [temp1, temp2, temp3, temp4]
    image = ToBNW(fname, fx = 1, ksize = 5)

    H, W = image.shape[0:2]
    # start template matching, and look for different angles:
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_CCORR_NORMED

    scorelst, reslst, infolst = [], [], []
    score1, res1, info1 = match1(image, temp1, slow=0.7, shigh=1.3)
    scorelst.append(score1)
    reslst.append(res1)
    infolst.append(info1)
    score2, res2, info2 = match1(image, temp2, slow=0.7, shigh=1.3)
    scorelst.append(score2)
    reslst.append(res2)
    infolst.append(info2)
    score3, res3, info3 = match1(image, temp3, slow=0.7, shigh=1.2)
    scorelst.append(score3)
    reslst.append(res3)
    infolst.append(info3)
    score4, res4, info4 = match1(image, temp3, slow=0.7, shigh=1.2)
    scorelst.append(score4)
    reslst.append(res4)
    infolst.append(info4)

    idx = np.array(scorelst).argmax()
    print(scorelst, idx)
    print(infolst[idx][:3])
    h, w = infolst[idx][3].shape[0:2]

    loc = np.unravel_index(reslst[idx].argmax(), reslst[idx].shape)
    print(loc)
    # print(info[3].shape, 'temp w and h', w, h, 'image W and H', W, H)
    # # # print(info)
    # get the corresponding window
    cornerr = loc[0]
    cornerc = loc[1]
    image_out = np.copy(image)
    image_out = cv2.circle(image_out, (cornerc, cornerr), radius=5, color=(0, 0, 255), thickness=1)
    image_out = cv2.circle(image_out, (cornerc + w, cornerr + h), radius=5, color=(0, 0, 255), thickness=-1)

    # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    # ax1.imshow(image_out, cmap=plt.cm.gray)
    # # ax2.plot(snake1[:, 1], snake1[:, 0], '.b', lw=1)
    # ax2.imshow(snake1[:, 1], snake1[:, 0], '.b', lw=1)
    # ax3.imshow(rmask, cmap=plt.cm.gray)
    # ax4.plot(snake2[:, 1], snake2[:, 0], '.b', lw=1)
    #
    # plt.show()


if __name__ == '__main__':
    # get_template()
    # rotation('../data/CC Lake slice/DSC00840.jpg')
    # ToBNW('../data/CC Lake slice/DSC00844.jpg', fx = 0.5)
    # AllBNW()
    # slice()
    # test()
    # rotate48('../data/CC Lake slice/DSC00848.jpg')
    display('../data/CC Lake slice/DSC00902.jpg')
