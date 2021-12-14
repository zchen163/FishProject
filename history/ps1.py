import math
import numpy as np
import cv2
import sys

# # Implement the functions below.


def extract_red(image):
    """ Returns the red channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the red channel.
    """
    temp_image = np.copy(image)
    redlayer = temp_image[:, :, 2]
    return redlayer


def extract_green(image):
    """ Returns the green channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the green channel.
    """
    temp_image =  np.copy(image)
    greenlayer = temp_image[:, :, 1]
    return greenlayer


def extract_blue(image):
    """ Returns the blue channel of the input image. It is highly recommended to make a copy of the
    input image in order to avoid modifying the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 2D array containing the blue channel.
    """
    temp_image = np.copy(image)
    bluelayer = temp_image[:, :, 0]
    return bluelayer

def swap_green_blue(image):
    """ Returns an image with the green and blue channels of the input image swapped. It is highly
    recommended to make a copy of the input image in order to avoid modifying the original array.
    You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input RGB (BGR in OpenCV) image.

    Returns:
        numpy.array: Output 3D array with the green and blue channels swapped.
    """
    temp_image = np.copy(image)
    temp_image[:, :, 0] = image[:, :, 1]
    temp_image[:, :, 1] = image[:, :, 0]
    return temp_image


def copy_paste_middle(src, dst, shape):
    """ Copies the middle region of size shape from src to the middle of dst. It is
    highly recommended to make a copy of the input image in order to avoid modifying the
    original array. You can do this by calling:
    temp_image = np.copy(image)

        Note: Assumes that src and dst are monochrome images, i.e. 2d arrays.

        Note: Where 'middle' is ambiguous because of any difference in the oddness
        or evenness of the size of the copied region and the image size, the function
        rounds downwards.  E.g. in copying a shape = (1,1) from a src image of size (2,2)
        into an dst image of size (3,3), the function copies the range [0:1,0:1] of
        the src into the range [1:2,1:2] of the dst.

    Args:
        src (numpy.array): 2D array where the rectangular shape will be copied from.
        dst (numpy.array): 2D array where the rectangular shape will be copied to.
        shape (tuple): Tuple containing the height (int) and width (int) of the section to be
                       copied.

    Returns:
        numpy.array: Output monochrome image (2D array)
    """
    temp_image = np.copy(src)
    print('Replace image')
    print(src, src.shape)
    print(dst, dst.shape)
    src_start_r = int(src.shape[0]/2 - shape[0]/2)
    src_end_r = src_start_r + shape[0]
    src_start_c = int(src.shape[1]/2 - shape[1]/2)
    src_end_c = src_start_c + shape[1]
    print(src_start_r, src_end_r, src_start_c, src_end_c)
    patch = temp_image[src_start_r:src_end_r, src_start_c:src_end_c]
    # print(patch)
    temp_image1 = np.copy(dst)
    dst_start_r = int(dst.shape[0]/2 - shape[0]/ 2)
    dst_end_r = dst_start_r + shape[0]
    dst_start_c = int(dst.shape[1]/2 - shape[1]/ 2)
    dst_end_c = dst_start_c + shape[1]
    print(dst_start_r, dst_end_r, dst_start_c, dst_end_c)
    temp_image1[dst_start_r:dst_end_r, dst_start_c:dst_end_c] = patch
    return temp_image1


def image_stats(image):
    """ Returns the tuple (min,max,mean,stddev) of statistics for the input monochrome image.
    In order to become more familiar with Numpy, you should look for pre-defined functions
    that do these operations i.e. numpy.min.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.

    Returns:
        tuple: Four-element tuple containing:
               min (float): Input array minimum value.
               max (float): Input array maximum value.
               mean (float): Input array mean / average value.
               stddev (float): Input array standard deviation.
    """
    temp_image = np.copy(image)
    temp_image = temp_image * 1.0 # convert to float
    # print(temp_image.dtype)
    min = np.amin(temp_image)
    max = np.amax(temp_image)
    mean = np.mean(temp_image)
    std = np.std(temp_image)
    # print(temp_image, temp_image.shape)
    # print(min, max, mean, std)
    return min, max, mean, std


def center_and_normalize(image, scale):
    """ Returns an image with the same mean as the original but with values scaled about the
    mean so as to have a standard deviation of "scale".

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        scale (int or float): scale factor.

    Returns:
        numpy.array: Output 2D image.
    """
    temp_image = np.copy(image)
    # print(temp_image)
    temp_image = temp_image * 1.0 # convert to float
    mean = np.mean(temp_image)
    std = np.std(temp_image)
    result = (temp_image - mean) / std * scale + mean
    # print(result)
    return result


def shift_image_left(image, shift):
    """ Outputs the input monochrome image shifted shift pixels to the left.

    The returned image has the same shape as the original with
    the BORDER_REPLICATE rule to fill-in missing values.  See

    http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/copyMakeBorder/copyMakeBorder.html?highlight=copy

    for further explanation.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): Input 2D image.
        shift (int): Displacement value representing the number of pixels to shift the input image.
            This parameter may be 0 representing zero displacement.

    Returns:
        numpy.array: Output shifted 2D image.
    """
    temp_image = np.copy(image)
    # print(temp_image.shape)
    shifted = temp_image[:, shift:]
    border = cv2.copyMakeBorder(shifted, 0, 0, 0, shift, borderType = cv2.BORDER_REPLICATE)
    # print(shifted.shape, border.shape)
    return border

def difference_image(img1, img2):
    """ Returns the difference between the two input images (img1 - img2). The resulting array must be normalized
    and scaled to fit [0, 255].

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        img1 (numpy.array): Input 2D image.
        img2 (numpy.array): Input 2D image.

    Returns:
        numpy.array: Output 2D image containing the result of subtracting img2 from img1.
    """
    temp_image1 = np.copy(img1) * 1.0
    temp_image2 = np.copy(img2) * 1.0
    # if not changing dtype to float, the difference under uint8 will be biased (based on course video)
    diff = temp_image1 - temp_image2
    # print(diff)
    normed = np.zeros(diff.shape)
    # print(normed)
    normed = cv2.normalize(diff, normed, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX)
    # print(normed)
    return normed


def add_noise(image, channel, sigma):
    """ Returns a copy of the input color image with Gaussian noise added to
    channel (0-2). The Gaussian noise mean must be zero. The parameter sigma
    controls the standard deviation of the noise.

    The returned array values must not be clipped or normalized and scaled. This means that
    there could be values that are not in [0, 255].

    Note: This function makes no defense against the creation
    of out-of-range pixel values.  Consider converting the input image to
    a float64 type before passing in an image.

    It is highly recommended to make a copy of the input image in order to avoid modifying
    the original array. You can do this by calling:
    temp_image = np.copy(image)

    Args:
        image (numpy.array): input RGB (BGR in OpenCV) image.
        channel (int): Channel index value.
        sigma (float): Gaussian noise standard deviation.

    Returns:
        numpy.array: Output 3D array containing the result of adding Gaussian noise to the
            specified channel.
    """
    temp_image = np.copy(image)
    # print('channel is ')
    # print(channel)
    layer = temp_image[:, :, channel]
    # print(layer, layer.shape)
    noise = np.random.randn(layer.shape[0], layer.shape[1]) * sigma
    # print(noise, noise.shape)
    noisy_layer = layer + noise
    # print(noisy_layer, noisy_layer.shape)
    temp_image[:, :, channel] = noisy_layer
    # print(temp_image, temp_image.shape)
    return temp_image
