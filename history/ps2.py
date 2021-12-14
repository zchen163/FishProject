"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2

import numpy as np


# use this function to find intersection(vertices) of lines. ref: https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def find_vertex(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0: return 0,0
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    # convert to monochrome
    image_in = np.copy(img_in)
    thresh1 = 70 # set threshold to 70/20 can show 3 circle
    thresh2 = 20

    # another way of finding the center: use yellow color mask
    lower_yellow = np.array([0,100,100])
    upper_yellow = np.array([20,255,255])
    mask = cv2.inRange(image_in, lower_yellow, upper_yellow)
    edge = cv2.Canny(mask, thresh1, thresh2)
    #dp = 1, minDist = 20, param1 = 100 or 50, param2 = 22,  minRadius = 10, maxRadius = 40
    center_circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp = 1, minDist = radii_range[0]*2, param1 = 50, param2 = 20,  minRadius = radii_range[0], maxRadius = radii_range[-1]+6)
    if center_circles is None:
        return (0,0), 'nothing'
    # for i in center_circles[0, :]:
        # print(i)
        # cv2.circle(image_in,(i[0],i[1]),i[2],(0,0,0),2)
        # cv2.circle(image_in,(i[0],i[1]),2,(0,0,0),3)

    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # identify the center of traffic light, and also find a up/bottom spot to determine color
    center_x = center_circles[0, :][0][0]
    center_y = center_circles[0, :][0][1]
    radius = center_circles[0, :][0][2]
    # print(center_x, center_y, radius)
    top_y = center_y - 2.5 * radius
    bottom_y = center_y + 2.5 * radius

    # determine color, remember the row and column vs x and y issue!
    color = None
    if image_in[int(top_y), int(center_x), 1] < 150 and image_in[int(top_y), int(center_x), 2] > 250:
        color = 'red'
    elif image_in[int(center_y), int(center_x), 1] > 250 and image_in[int(center_y), int(center_x), 2] > 250:
        color = 'yellow'
    elif image_in[int(bottom_y), int(center_x), 1] > 250 and image_in[int(bottom_y), int(center_x), 2] < 150:
        color = 'green'
    return (center_x, center_y), color


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    # print('start yield Sign detection')
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 5)
    lower_red = np.array([0,0,140])
    upper_red = np.array([20,20,255])
    mask = cv2.inRange(image_in, lower_red, upper_red)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)

    # detection1: lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*30, threshold = 50, minLineLength = 70, maxLineGap = 4)
    # lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*30, threshold = 65, minLineLength = 90, maxLineGap = 5)
    # lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*30, threshold = 60, minLineLength = 70, maxLineGap = 5) works sometime on grader
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*30, threshold = 50, minLineLength = 60, maxLineGap = 5)
    if lines is None:
        return (0,0)
    side_top = np.empty([0,4])
    side_left = np.empty([0,4])
    side_right = np.empty([0,4])
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        # cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        theta = (y2-y1)/(x2-x1)
        # print(theta)
        if theta == 0:
            side_top = np.append(side_top, i, axis = 0)
            # print('top')
        elif theta < 0 and theta > -1.8:
            side_right = np.append(side_right, i, axis = 0)
            # print('right')
        elif theta > 0 and theta < 1.8:
            side_left = np.append(side_left, i, axis = 0)
            # print('left')
    if side_left.shape[0] == 0 or side_right.shape[0] == 0:
        return (0,0)
    # get 3 sides that outside of the yield sign:
    side_top1 = side_top[side_top[:,1].argsort()]
    side_left1 = side_left[side_left[:,0].argsort()]
    side_right1 = side_right[side_right[:,3].argsort()]
    # print('after sort')
    topline = side_top1[0, :].astype(np.int)
    leftline =side_left1[0, :].astype(np.int)
    rightline =side_right1[0, :].astype(np.int)
    # print(leftline)
    cv2.line(image_in,(topline[0],topline[1]),(topline[2], topline[3]),(0,0,0),2)
    cv2.line(image_in,(leftline[0],leftline[1]),(leftline[2], leftline[3]),(0,0,0),2)
    cv2.line(image_in,(rightline[0],rightline[1]),(rightline[2], rightline[3]),(0,0,0),2)
    a = topline[0:2]
    b = topline[2:4]
    c = leftline[0:2]
    d = leftline[2:4]
    e = rightline[0:2]
    f = rightline[2:4]
    vertex1 = find_vertex((a,b),(c,d))
    vertex2 = find_vertex((a,b),(e,f))
    vertex3 = find_vertex((c,d),(e,f))
    # cv2.circle(image_in, vertex1, 3, (0,0,0), -1)
    # cv2.circle(image_in, vertex2, 3, (0,0,0), -1)
    # cv2.circle(image_in, vertex3, 3, (0,0,0), -1)
    center_x = int((vertex1[0] + vertex2[0] + vertex3[0])/3)
    center_y = int((vertex1[1] + vertex2[1] + vertex3[1])/3)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)

    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)

def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    # print('start stop Sign detection')
    image_in = np.copy(img_in)
    # use median filter to blur the text
    filtered = cv2.medianBlur(image_in, 7)
    # apply a red color mask:
    lower_red = np.array([0,0,120])
    upper_red = np.array([20,20,255])
    mask = cv2.inRange(filtered, lower_red, upper_red)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)

    # find vertical lines using probalistic houghline transformation, thres 30 minline 25 maxlin 4 works
    # stop1: lines2 = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*45, threshold = 15, minLineLength = 25, maxLineGap = 8)
    # lines2 = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180, threshold = 30, minLineLength = 25, maxLineGap = 4)
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*45, threshold = 15, minLineLength = 20, maxLineGap = 4)
    if lines is None:
        return (0,0)
    # sides = np.copy(lines)
    sides1 = np.empty([0,4])
    sides2 = np.empty([0,4])
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        # find 45 degree sides
        b = x2-x1
        a = y2-y1
        theta = np.nan_to_num(a/ b)
        # print(theta)
        if theta > 0.8 and theta < 1.2:
            # print(sides, sides.shape, i, i.shape)
            sides1 = np.append(sides1, i, axis = 0) # side1 topleft to bottomright
        elif theta > -1.2 and theta < -0.8:
            sides2 = np.append(sides2, i, axis = 0) # side2 topright to bottomleft
    # print('45 degree sides: ')
    # print(sides1, sides2)
    if sides1.shape[0] == 0 or sides2.shape[0] == 0:
        return (0,0)
    # identify 4 most outside sides
    side1 = sides1[sides1[:,0].argsort()][0, :].astype(np.int) # bottomleft side
    side2 = sides1[sides1[:,1].argsort()][0, :].astype(np.int) # topright side
    side3 = sides2[sides2[:,0].argsort()][0, :].astype(np.int) # topleft side
    side4 = sides2[sides2[:,2].argsort()][-1, :].astype(np.int) # topleft side
    # print(side4)
    # get vertices
    vertex1 = find_vertex((side1[0:2],side1[2:4]),(side3[0:2],side3[2:4]))
    vertex2 = find_vertex((side1[0:2],side1[2:4]),(side4[0:2],side4[2:4]))
    vertex3 = find_vertex((side2[0:2],side2[2:4]),(side3[0:2],side3[2:4]))
    vertex4 = find_vertex((side2[0:2],side2[2:4]),(side4[0:2],side4[2:4]))
    # cv2.circle(image_in, vertex1, 3, (0,0,0), -1)
    # cv2.circle(image_in, vertex2, 3, (0,0,0), -1)
    # cv2.circle(image_in, vertex3, 3, (0,0,0), -1)
    # cv2.circle(image_in, vertex4, 3, (0,0,0), -1)

    center_x = int((vertex1[0] + vertex2[0] + vertex3[0] + vertex4[0])/4)
    center_y = int((vertex1[1] + vertex2[1] + vertex3[1] + vertex4[1])/4)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 7)
    lower_yellow = np.array([0,240,240])
    upper_yellow = np.array([10,255,255])
    mask = cv2.inRange(filtered, lower_yellow, upper_yellow)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    # lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*45, threshold = 34, minLineLength = 28, maxLineGap = 2)
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*45, threshold = 22, minLineLength = 30, maxLineGap = 2)
    if lines is None:
        return (0,0)
    lines = lines[0:4, :]

    # pick top 4 lines:
    x = []
    y = []
    dist = []
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        x = np.append(x, x1)
        x = np.append(x, x2)
        y = np.append(y, y1)
        y = np.append(y, y2)
    # find most top, most left, most right and most down point:
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    center_x = int(np.mean([xmin, xmax]))
    center_y = int(np.mean([ymin, ymax]))
    # print(center_x, center_y)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)

    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    # print('start construction Sign detection')
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 5)
    lower_orange = np.array([0,100,230])
    upper_orange = np.array([30,150,255])
    # image_HSV = cv.cvtColor(image_in, cv.COLOR_BGR2HSV)
    mask = cv2.inRange(image_in, lower_orange, upper_orange)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    # cv2.imshow('a', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180 * 45, threshold = 47, minLineLength = 38, maxLineGap = 2)
    if lines is None:
        return (0,0)
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)

    sides = np.copy(lines)
    sides1 = sides[abs(sides[:,:,1]-sides[:,:,3]) / abs(sides[:,:,0]-sides[:,:,2]) < 1.10]
    condition = abs((sides1[:, 1] - sides1[:, 3])/ (sides1[:,0]-sides1[:,2])) > 0.9
    sides2 = sides1[condition, :]
    # print(sides2)

    center_x = int(np.mean((sides2[:, 0] + sides2[:, 2])/2))
    center_y = int(np.mean((sides2[:, 1] + sides2[:, 3])/2))
    # print(center_x, center_y)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)
    return (center_x, center_y)


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    # print('start Do Not Enter Sign detection')
    image_in = np.copy(img_in)
    # apply a red color mask:
    lower_red = np.array([0,0,100])
    upper_red = np.array([10,10,255])
    mask = cv2.inRange(image_in, lower_red, upper_red)
    thresh1 = 70 # set threshold to 70/20 can show 3 circle
    thresh2 = 20
    edge = cv2.Canny(mask, thresh1, thresh2)
    circle = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp = 1, minDist = 20, param1 = 100, param2 = 25,  minRadius = 20, maxRadius = 60)
    if circle is None: return (0,0)
    # identify the sign by: center of the sign is white, and a little on top of the center is red:
    for i in circle[0, :]:
        # print(image_in.shape)
        centerpoint = image_in[int(i[1]), int(i[0]), :]
        up_point = image_in[int(i[1] - i[2]/2), int(i[0]), :]
        if centerpoint[0] > 250 and centerpoint[1] > 250 and centerpoint[2] > 250:
            if up_point[0] < 10 and up_point[1] < 10 and up_point[2] > 250:
                center_y = i[1]
                center_x = i[0]
        # cv2.circle(image_in,(i[0],i[1]),i[2],(0,255,0),2)
        # cv2.circle(image_in,(i[0],i[1]),2,(0,0,255),3)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (center_x, center_y)


def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    result = {}
    image_in = np.copy(img_in)
    tl, color = traffic_light_detection(image_in, range(4, 25, 1))
    if tl != (0,0):
    	result['traffic_light'] = tl

    dne = do_not_enter_sign_detection(image_in)
    if dne != (0,0):
    	result['no_entry'] = dne

    stop = stop_sign_detection(image_in)
    if stop != (0,0):
    	result['stop'] = stop

    warning = warning_sign_detection(image_in)
    if warning != (0,0):
    	result['warning'] = warning

    yield_sign = yield_sign_detection(image_in)
    if yield_sign != (0,0):
    	result['yield'] = yield_sign

    constr = construction_sign_detection(image_in)
    if constr != (0,0):
    	result['construction'] = constr
    # print(result)
    return result

def construction_sign_detection1(img_in):
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 5)
    lower_orange = np.array([0,100,230])
    upper_orange = np.array([30,150,255])
    mask = cv2.inRange(image_in, lower_orange, upper_orange)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180 * 45, threshold = 20, minLineLength = 45, maxLineGap = 8)
    if lines is None:
        return (0,0)
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
    sides = np.copy(lines)
    sides1 = sides[abs(sides[:,:,1]-sides[:,:,3]) / abs(sides[:,:,0]-sides[:,:,2]) < 1.10]
    condition = abs((sides1[:, 1] - sides1[:, 3])/ (sides1[:,0]-sides1[:,2])) > 0.9
    sides2 = sides1[condition, :]
    center_x = int(np.mean((sides2[:, 0] + sides2[:, 2])/2))
    center_y = int(np.mean((sides2[:, 1] + sides2[:, 3])/2))
    # print(center_x, center_y)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)


def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    # print('starting denoising')
    image_in = np.copy(img_in)
    # denoise
    filtered = cv2.medianBlur(image_in, 7)
    denoised = cv2.fastNlMeansDenoisingColored(filtered, None, 12, 12, 7, 21)
    # cv2.imshow('a', denoised)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('start detecting tl')
    result = {}
    tl, color = traffic_light_detection(denoised, range(4, 25, 1))
    if tl != (0,0):
    	result['traffic_light'] = tl

    dne = do_not_enter_sign_detection(denoised)
    if dne != (0,0):
    	result['no_entry'] = dne

    stop = stop_sign_detection(filtered)
    if stop != (0,0):
    	result['stop'] = stop

    warning = warning_sign_detection(denoised)
    if warning != (0,0):
    	result['warning'] = warning

    yield_sign = yield_sign_detection(denoised)
    if yield_sign != (0,0):
    	result['yield'] = yield_sign

    constr = construction_sign_detection1(denoised)
    if constr != (0,0):
    	result['construction'] = constr
    return result

def do_not_enter_sign_detection_real(img_in):
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 1)
    lower_red = np.array([0,0,100])
    upper_red = np.array([50,50,255])
    mask = cv2.inRange(filtered, lower_red, upper_red)
    edge = cv2.Canny(mask, 70, 20)
    # cv2.imshow('a', edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    circle = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp = 1, minDist = 200, param1 = 100, param2 = 25,  minRadius = 10, maxRadius = 200)
    if circle is None: return (0,0)
    for i in circle[0, :]:

        centerpoint = image_in[int(i[1]), int(i[0]), :]
        up_point = image_in[int(i[1] - i[2]/2), int(i[0]), :]
        if centerpoint[0] > 230 and centerpoint[1] > 230 and centerpoint[2] > 230:
            center_y = i[1]
            center_x = i[0]
        else: return (0,0)
        cv2.circle(image_in,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(image_in,(i[0],i[1]),2,(0,0,255),3)
    return (center_x, center_y)

def stop_sign_detection_real(img_in):
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 7)
    # apply a red color mask:
    lower_red = np.array([0,0,100])
    upper_red = np.array([70,70,255])
    mask = cv2.inRange(filtered, lower_red, upper_red)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*45, threshold = 15, minLineLength = 30, maxLineGap = 4)
    if lines is None:
        return (0,0)
    sides1 = np.empty([0,4])
    sides2 = np.empty([0,4])
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        # find 45 degree sides
        b = x2-x1
        a = y2-y1
        theta = np.nan_to_num(a/ b)
        # print(theta)
        if theta > 0.8 and theta < 1.2:
            # print(sides, sides.shape, i, i.shape)
            sides1 = np.append(sides1, i, axis = 0) # side1 topleft to bottomright
        elif theta > -1.2 and theta < -0.8:
            sides2 = np.append(sides2, i, axis = 0) # side2 topright to bottomleft
    if sides1.shape[0] == 0 or sides2.shape[0] == 0:
        return (0,0)
    # identify 4 most outside sides
    side1 = sides1[sides1[:,0].argsort()][0, :].astype(np.int) # bottomleft side
    side2 = sides1[sides1[:,1].argsort()][0, :].astype(np.int) # topright side
    side3 = sides2[sides2[:,0].argsort()][0, :].astype(np.int) # topleft side
    side4 = sides2[sides2[:,2].argsort()][-1, :].astype(np.int) # topleft side
    # get vertices
    vertex1 = find_vertex((side1[0:2],side1[2:4]),(side3[0:2],side3[2:4]))
    vertex2 = find_vertex((side1[0:2],side1[2:4]),(side4[0:2],side4[2:4]))
    vertex3 = find_vertex((side2[0:2],side2[2:4]),(side3[0:2],side3[2:4]))
    vertex4 = find_vertex((side2[0:2],side2[2:4]),(side4[0:2],side4[2:4]))
    cv2.circle(image_in, vertex1, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex2, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex3, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex4, 3, (0,0,0), -1)

    center_x = int((vertex1[0] + vertex2[0] + vertex3[0] + vertex4[0])/4)
    center_y = int((vertex1[1] + vertex2[1] + vertex3[1] + vertex4[1])/4)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)
    # cv2.imshow('a', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)

def yield_sign_detection_real(img_in):
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 5)
    lower_red = np.array([0,0,140])
    upper_red = np.array([50,50,255])
    mask = cv2.inRange(image_in, lower_red, upper_red)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    # cv2.imshow('a', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180*30, threshold = 50, minLineLength = 60, maxLineGap = 5)
    if lines is None: return (0,0)
    side_top = np.empty([0,4])
    side_left = np.empty([0,4])
    side_right = np.empty([0,4])
    for i in lines:
        # print(i)
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        # cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        theta = (y2-y1)/(x2-x1)
        if theta == 0:
            side_top = np.append(side_top, i, axis = 0)
        elif theta < 0 and theta > -1.8:
            side_right = np.append(side_right, i, axis = 0)
        elif theta > 0 and theta < 1.8:
            side_left = np.append(side_left, i, axis = 0)
    if side_left.shape[0] == 0 or side_right.shape[0] == 0: return (0,0)
    # get 3 sides that outside of the yield sign:
    side_top1 = side_top[side_top[:,1].argsort()]
    side_left1 = side_left[side_left[:,0].argsort()]
    side_right1 = side_right[side_right[:,3].argsort()]
    # print('after sort')
    topline = side_top1[0, :].astype(np.int)
    leftline =side_left1[0, :].astype(np.int)
    rightline =side_right1[0, :].astype(np.int)
    # print(leftline)
    cv2.line(image_in,(topline[0],topline[1]),(topline[2], topline[3]),(0,0,0),2)
    cv2.line(image_in,(leftline[0],leftline[1]),(leftline[2], leftline[3]),(0,0,0),2)
    cv2.line(image_in,(rightline[0],rightline[1]),(rightline[2], rightline[3]),(0,0,0),2)
    a = topline[0:2]
    b = topline[2:4]
    c = leftline[0:2]
    d = leftline[2:4]
    e = rightline[0:2]
    f = rightline[2:4]
    vertex1 = find_vertex((a,b),(c,d))
    vertex2 = find_vertex((a,b),(e,f))
    vertex3 = find_vertex((c,d),(e,f))
    center_x = int((vertex1[0] + vertex2[0] + vertex3[0])/3)
    center_y = int((vertex1[1] + vertex2[1] + vertex3[1])/3)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)
    return (center_x, center_y)

def warning_sign_detection_real(img_in):
    image_in = np.copy(img_in)
    filtered = cv2.medianBlur(image_in, 15)
    lower_yellow = np.array([0,190,190])
    upper_yellow = np.array([70,255,255])
    mask = cv2.inRange(filtered, lower_yellow, upper_yellow)
    thresh1 = 10
    thresh2 = 0
    edge = cv2.Canny(mask, thresh1, thresh2)
    lines = cv2.HoughLinesP(edge, rho = 1, theta = np.pi/180, threshold = 30, minLineLength = 20, maxLineGap = 5)
    if lines is None: return (0,0)
    lines = lines[0:4, :]
    # pick top 4 lines:
    sides1 = np.empty([0,4])
    sides2 = np.empty([0,4])
    for i in lines:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[0][2]
        y2 = i[0][3]
        cv2.line(image_in,(x1,y1),(x2, y2),(0,0,0),2)
        # find 45 degree sides
        b = x2-x1
        a = y2-y1
        theta = np.nan_to_num(a/ b)
        if theta > 0.8 and theta < 1.2:
            sides1 = np.append(sides1, i, axis = 0) # side1 topleft to bottomright
        elif theta > -1.2 and theta < -0.8:
            sides2 = np.append(sides2, i, axis = 0) # side2 topright to bottomleft
    if sides1.shape[0] == 0 or sides2.shape[0] == 0:
        return (0,0)
    # identify 4 most outside sides
    side1 = sides1[sides1[:,0].argsort()][0, :].astype(np.int) # bottomleft side
    side2 = sides1[sides1[:,1].argsort()][0, :].astype(np.int) # topright side
    side3 = sides2[sides2[:,0].argsort()][0, :].astype(np.int) # topleft side
    side4 = sides2[sides2[:,2].argsort()][-1, :].astype(np.int) # topleft side
    # get vertices
    vertex1 = find_vertex((side1[0:2],side1[2:4]),(side3[0:2],side3[2:4]))
    vertex2 = find_vertex((side1[0:2],side1[2:4]),(side4[0:2],side4[2:4]))
    vertex3 = find_vertex((side2[0:2],side2[2:4]),(side3[0:2],side3[2:4]))
    vertex4 = find_vertex((side2[0:2],side2[2:4]),(side4[0:2],side4[2:4]))
    cv2.circle(image_in, vertex1, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex2, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex3, 3, (0,0,0), -1)
    cv2.circle(image_in, vertex4, 3, (0,0,0), -1)

    center_x = int((vertex1[0] + vertex2[0] + vertex3[0] + vertex4[0])/4)
    center_y = int((vertex1[1] + vertex2[1] + vertex3[1] + vertex4[1])/4)
    cv2.circle(image_in, (center_x, center_y), 3, (0,0,0), -1)
    # cv2.imshow('b', image_in)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return (center_x, center_y)

def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    '''image source:
    1. https://www.mynrma.com.au/cars-and-driving/driver-training-and-licences/resources/does-the-three-seconds-stop-rule-exist
    2.
    https://arslocii.files.wordpress.com/2010/09/krimpet-yield.jpeg
    3. https://images.unsplash.com/photo-1573133351397-7eb7d3d7b3dc?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80
    4.
    https://dmzn2b8hkpq8b.cloudfront.net/images/products/515x515/S294425.jpg
    5.
    https://www.trafficsigns.com/media/catalog/category/Regulatory_Page_3.jpg
    6.
    https://secure.img1-fg.wfcdn.com/im/37319650/resize-h600-w600%5Ecompr-r85/1406/14062831/Street+Signs+Fabric+Wall+Sticker.jpg

    all jpg files were transferred into png by website: https://jpg2png.com/
    '''
    # print('starting real image')
    image_in = np.copy(img_in)
    # denoise
    # filtered = cv2.medianBlur(image_in, 7)
    denoised = cv2.fastNlMeansDenoisingColored(image_in, None, 12, 12, 7, 21)
    # cv2.imshow('a', denoised)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print('start detecting tl')
    result = {}
    tl, color = traffic_light_detection(denoised, range(4, 20, 1))
    if tl != (0,0):
    	result['traffic_light'] = tl

    dne = do_not_enter_sign_detection_real(denoised)
    if dne != (0,0):
    	result['no_entry'] = dne

    stop = stop_sign_detection_real(denoised)
    if stop != (0,0):
    	result['stop'] = stop

    warning = warning_sign_detection_real(denoised)
    if warning != (0,0):
    	result['warning'] = warning

    yield_sign = yield_sign_detection_real(denoised)
    if yield_sign != (0,0):
    	result['yield'] = yield_sign

    constr = construction_sign_detection(denoised)
    if constr != (0,0):
    	result['construction'] = constr
    return result
    # raise NotImplementedError
