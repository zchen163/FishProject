"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
from scipy.ndimage import rotate

def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    x1 = p0[0]
    y1 = p0[1]
    x2 = p1[0]
    y2 = p1[1]
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    image_in = np.copy(image)
    h = image_in.shape[0]
    w = image_in.shape[1]
    topleft = (0, 0)
    botleft = (0, h-1)
    topright = (w-1, 0)
    botright = (w-1, h-1)
    # cv2.circle(image_in, topleft, 3, (0,0,255), -1)
    # cv2.circle(image_in, botleft, 3, (0,0,255), -1)
    # cv2.circle(image_in, topright, 3, (0,0,255), -1)
    # cv2.circle(image_in, botright, 3, (0,0,255), -1)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return [topleft, botleft, topright, botright]

def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    image_in = np.copy(image)
    temp_in = np.copy(template)
    # cv2.imshow('b', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # try convert to gray by a color mask
    lower_yellow = np.array([50,50,100])
    upper_yellow = np.array([255,255,255])
    gray = cv2.inRange(image_in, lower_yellow, upper_yellow)

    # cv2.imshow('b', gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    temp_gray = cv2.cvtColor(temp_in, cv2.COLOR_BGR2GRAY)
    # start template matching, and look for different angles:
    best_match_score = 0
    # all available methods: methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = cv2.TM_CCOEFF_NORMED
    for i in range(36):
        # the rotate takes angle in degrees
        temp_rot = rotate(temp_gray, 5 * i, mode = 'constant', reshape = False, cval = 0)
        # set reshape = true, cval = 255 good for part 4-1,2,4...
        res_rot = cv2.matchTemplate(gray, temp_rot, method)
        if res_rot.max() >= best_match_score:
            best_match_score = res_rot.max()
            res = res_rot
            best_temp = temp_rot
    temp_w = best_temp.shape[1]
    temp_h = best_temp.shape[0]
    # cv2.imshow('b', res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # setup threshold to pick all values
    threshold = 0.4
    loc = np.where(res >= threshold) # np.where returns row and column, rather x and y
    # print(loc)
    xlst = loc[1]
    ylst = loc[0]

    # start a loop to find 4 highest matched point:
    centers = []
    radius = 10
    # print(xlst, xlst.shape)

    while len(centers) < 4:
        (minval, maxval, minloc, maxloc) = cv2.minMaxLoc(res)
        centers.append(maxloc)
        for i in range(len(xlst)):
            dist = euclidean_distance(maxloc, (xlst[i], ylst[i]))
            if dist <= radius: # make all around area to 0, and radius is defined above
                res[ylst[i], xlst[i]] = 0
    # to determine the location, first sort the point by x axis:
    # reference: https://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value
    centers = sorted(centers, key=lambda x: x[0])
    # define the topleft, botleft, and restore the real location by adding half width and height of the template
    if centers[0][1] <= centers[1][1]:
        topleft = (centers[0][0] + temp_w//2, centers[0][1] + temp_h//2)
        botleft = (centers[1][0] + temp_w//2, centers[1][1] + temp_h//2)
    else:
        topleft = (centers[1][0] + temp_w//2, centers[1][1] + temp_h//2)
        botleft = (centers[0][0] + temp_w//2, centers[0][1] + temp_h//2)
    if centers[2][1] <= centers[3][1]:
        topright = (centers[2][0] + temp_w//2, centers[2][1] + temp_h//2)
        botright = (centers[3][0] + temp_w//2, centers[3][1] + temp_h//2)
    else:
        topright = (centers[3][0] + temp_w//2, centers[3][1] + temp_h//2)
        botright = (centers[2][0] + temp_w//2, centers[2][1] + temp_h//2)

    # cv2.circle(image_in, topleft, 3, (0,0,255), -1)
    # cv2.circle(image_in, botleft, 3, (0,0,255), -1)
    # cv2.circle(image_in, topright, 3, (0,0,255), -1)
    # cv2.circle(image_in, botright, 3, (0,0,255), -1)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(topleft, botleft, topright, botright)
    return [topleft, botleft, topright, botright]


def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    image_in = np.copy(image)
    # topleft-botleft
    cv2.line(image_in, (markers[0][0], markers[0][1]), (markers[1][0], markers[1][1]), color= (0,0,255), thickness = thickness)
    # topleft-topright
    cv2.line(image_in, (markers[0][0], markers[0][1]), (markers[2][0], markers[2][1]), color= (0,0,255), thickness = thickness)
    # topright-botright
    cv2.line(image_in, (markers[2][0], markers[2][1]), (markers[3][0], markers[3][1]), color= (0,0,255), thickness = thickness)
    # botleft-botright
    cv2.line(image_in, (markers[1][0], markers[1][1]), (markers[3][0], markers[3][1]), color= (0,0,255), thickness = thickness)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_in

def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    # test the projection by the example (2D) in OH:
    # src_x = np.zeros((2, 5))
    # print(src_x)
    # dst_x = np.zeros((2, 15))
    # dst_x[0, :] = np.arange(15)
    # dst_x[1, :] = 1
    # print(dst_x)
    # H = np.array([[0.5, -1], [0, 1]])
    # print(H)
    # src_pos = np.dot(H, dst_x)
    # a = src_pos[0, :] / src_pos[1, :]
    # print(a)
    # # ind = np.where((src_pos[0, :]>0) and (src_pos[0, :] < 6))
    # select = src_pos[:, (src_pos[0,:]>0) & (src_pos[0,:]<=5)].astype(np.int32)
    # print(select)

    H = np.copy(homography)
    Hni = np.linalg.inv(H)
    # print(Hni)
    image_A = np.copy(imageA) # source
    image_B = np.copy(imageB) # destination
    Ah = image_A.shape[0]
    Aw = image_A.shape[1]
    Bh = image_B.shape[0]
    Bw = image_B.shape[1]
    # print(Ah, Aw, Bh, Bw)
    # calculate a position corresponding matrix tha map pos B to A, use backward warping
    # each column of the dst_position matrix is [xd, yd, 1]T, then the calculated src_pos is in format of [xs*w, ys*w, w]T
    dst_pos = np.zeros((3, Bh * Bw), np.int32)
    dst_pos[2, :] = 1
    for i in range(Bh):
        dst_pos[0, i*Bw:(i+1)*Bw] = np.arange(Bw)
        dst_pos[1, i*Bw:(i+1)*Bw] = i
    # print(dst_pos, dst_pos.shape)
    # the src_pos matrix is calculates as: src_pos = H-1 * dst_pos
    src_pos = np.dot(Hni, dst_pos)
    src_pos = (src_pos/src_pos[2, :])[0:2, :]
    # then remove the value inside below 0 and above range
    dst_pos = dst_pos[0:2, :]
    idx = np.where((src_pos[0,:]>=0) & (src_pos[0,:]<=(Aw-1)) & (src_pos[1,:]>=0) & (src_pos[1,:]<=(Ah-1)))[0]
    dst_pos = dst_pos[:, idx].astype(np.int32)
    src_pos = src_pos[:, idx].astype(np.int32)
    # print(src_pos, src_pos.shape)
    # now map the projection:
    src_x = src_pos[0, :]
    src_y = src_pos[1, :]
    dst_x = dst_pos[0, :].astype(np.int32)
    dst_y = dst_pos[1, :].astype(np.int32)
    # print(image_B[dst_y, dst_x, :].shape)
    # print(image_A[src_y, src_x, :].shape)
    image_B[dst_y, dst_x, :] = image_A[src_y, src_x, :]
    # cv2.imshow('a', image_B)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image_B


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    src = np.copy(src_points)
    dst = np.copy(dst_points)
    # use AH = b
    H = np.zeros((3, 3))
    A = np.zeros((8, 8))
    b = np.zeros((8, 1))
    # construct A and b, then use it to solve H
    for i in range(len(src)):
        xs = src[i][0]
        ys = src[i][1]
        xd = dst[i][0]
        yd = dst[i][1]
        b[i*2, 0] = xd
        b[i*2+1, 0] = yd
        # xs, ys, 1, 0, 0, 0, -xsxd, -ysxd
        A[i*2, :] = np.asarray([xs, ys, 1, 0, 0, 0, -xs*xd, -ys*xd])
        # 0, 0, 0, xs, ys, 1, -xsyd, -ysyd
        A[i*2+1, :] = np.asarray([0, 0, 0, xs, ys, 1, -xs*yd, -ys*yd])
    # print(A)
    # print(b)
    x = np.linalg.lstsq(A, b)[0]
    # print(np.linalg.lstsq(A, b))
    H[0, :] = [x[0], x[1], x[2]]
    H[1, :] = [x[3], x[4], x[5]]
    H[2, :] = [x[6], x[7], 1]
    # print(H)

    # sanity check: (xd, yd, w)T = H * ()T
    # a1 = src.transpose()
    # a1 = np.append(a1, np.ones((1, 4)), axis = 0)
    # print(a1)
    # result = np.dot(H, a1)
    # print(result)
    return H


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame

        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
