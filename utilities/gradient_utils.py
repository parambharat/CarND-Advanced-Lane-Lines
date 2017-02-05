import numpy as np
import cv2
from .general_utils import gray_scale, norm

COLORS = ['HLS', 'HSV', 'LAB', 'LUV', 'XYZ', 'YUV']


def get_mask_vertices(imshape):
    mask_vertices = np.array(
        [[[imshape[1] * 0, imshape[0] * 1],
          [imshape[1] * 0.425, imshape[0] * 0.630],
          [imshape[1] * 0.575, imshape[0] * 0.630],
          [imshape[1] * 1, imshape[0] * 1]
          ]],
        dtype=np.int32)
    return mask_vertices


def get_single_channel(img, c_space='gray', channel=0):
    """Selects a single channel from a 3 channel image.
    By default returns the gray scale image
    Input:
        img: image to extract single channel from
        c_space: The color space name in cv2 format string.
            example 'HLS'
        channel: The channel to pick from the color space.
    Returns:
        single: Single channel Numpy Image array.
    """
    if c_space == 'gray':
        single = gray_scale(img)
    elif c_space in COLORS:
        c_space = '{}{}'.format('cv2.COLOR_RGB2', c_space)
        colored = cv2.cvtColor(img, eval(c_space))
        single = colored[:, :, channel]
    else:
        single = img[:, :, channel]
    return single


def apply_threshold(img, thresh=(0, 255)):
    thresholded = np.zeros_like(img)
    thresholded[(img > thresh[0]) & (img < thresh[1])] = 1
    return thresholded


def color_threshold(
        img, color='HLS', thresh=(0, 255), channel=2):
    """Applies color thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply color threshold on
        color: string, color space to transform the RGB image to
        thresh: tuple, (minimum, maximum) threshold values
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    if color in COLORS:
        color = '{}{}'.format('cv2.COLOR_RGB2', color)
        dst_img = cv2.cvtColor(img, eval(color))
    else:
        dst_img = img
    mask_channel = dst_img[:, :, channel]
    binary_mask = apply_threshold(mask_channel, thresh=thresh)

    return binary_mask


def combine_color_threshold(img, **kwargs):
    HLS_dict = kwargs['HLS']
    LAB_dict = kwargs['LAB']
    LUV_dict = kwargs['LUV']
    HLS_thresholded = color_threshold(img, **HLS_dict)
    LAB_thresholded = color_threshold(img, **LAB_dict)
    LUV_thresholded = color_threshold(img, **LUV_dict)
    combined = np.zeros_like(HLS_thresholded)
    combined[
        (HLS_thresholded == 1) |
        ((LAB_thresholded == 1) |
         (LUV_thresholded == 1))] = 1
    return combined


def abs_sobel_threshold(
        img, orient='x', kernel_size=5,
        thresh=(0, 255), c_space='gray', channel=0):
    """Applies absolute sobel thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply sobel threshold on
        orient: string, 'x' or 'y', orientation to apply soble transform
        kernel_size: size of the sobel kernel
        thresh: tuple, (minimum, maximum) threshold values
        c_space: string, color space to transform the RGB image to
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    single = get_single_channel(img, c_space, channel)

    if orient == 'x':
        soblet = cv2.Sobel(
            single, cv2.CV_64F, 1, 0, ksize=kernel_size)
    elif orient == 'y':
        soblet = cv2.Sobel(
            single, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_s = np.absolute(soblet)
    scaled = np.uint8(255 * abs_s / np.max(abs_s))

    binary_mask = apply_threshold(scaled, thresh=thresh)
    return binary_mask


def mag_threshold(
        img, kernel_size=3, thresh=(0, 255),
        c_space='gray', channel=0):
    """Applies magnitude sobel thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply sobel threshold on
        kernel_size: size of the sobel kernel
        thresh: tuple, (minimum, maximum) threshold values
        c_space: string, color space to transform the RGB image to
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    single = get_single_channel(img, c_space, channel)

    sobelx = cv2.Sobel(
        single, cv2.CV_64F, 1, 0, ksize=kernel_size)

    sobely = cv2.Sobel(
        single, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobelt = np.sqrt(sobelx**2 + sobely**2)
    scaled = np.uint8(255 * sobelt / np.max(sobelt))
    binary_mask = apply_threshold(scaled, thresh=thresh)
    return binary_mask


def dir_threshold(
        img, kernel_size=3, thresh=(0, np.pi / 2),
        c_space='gray', channel=0):
    """Applies direction sobel thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply sobel threshold on
        kernel_size: size of the sobel kernel
        thresh: tuple, (minimum, maximum) threshold values
        c_space: string, color space to transform the RGB image to
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    single = get_single_channel(img, c_space, channel)

    sobelx = cv2.Sobel(
        single, cv2.CV_64F, 1, 0, ksize=kernel_size)

    sobely = cv2.Sobel(
        single, cv2.CV_64F, 0, 1, ksize=kernel_size)

    abs_sx = np.absolute(sobelx)
    abs_sy = np.absolute(sobely)
    d_grad = np.arctan2(abs_sy, abs_sx)
    binary_mask = apply_threshold(d_grad, thresh=thresh)
    return binary_mask


def canny_threshold(
        img, kernel_size=7, thresh=(50, 100),
        c_space='gray', channel=0):
    """Applies canny thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply canny threshold on
        kernel_size: size of the gaussian kernel
        thresh: tuple, (minimum, maximum) threshold values
        c_space: string, color space to transform the RGB image to
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """

    single = get_single_channel(img, c_space, channel)

    blurred_img = cv2.GaussianBlur(
        single, (kernel_size, kernel_size), 0)
    canny_img = cv2.Canny(
        blurred_img, thresh[0], thresh[1])
    return canny_img


def laplace_threshold(
        img, ddepth=5, kernel_size=5,
        thresh=(0, 255), c_space='gray', channel=0):
    """Applies absolute sobel thresholding to a single channel of the image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply sobel threshold on
        ddepth: int size of depth to apply
        kernel_size: size of the sobel kernel
        thresh: tuple, (minimum, maximum) threshold values
        c_space: string, color space to transform the RGB image to
        channel: channel to apply threshold on
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    single = get_single_channel(img, c_space, channel)

    lap = cv2.Laplacian(single, ddepth=ddepth, ksize=kernel_size)

    abs_s = np.absolute(lap)
    scaled = np.uint8(255 * abs_s / np.max(abs_s))

    binary_mask = apply_threshold(scaled, thresh=thresh)
    return binary_mask


def apply_thresholds(img, **kwargs):
    """Applies various thresholding to an image.
    Inputs:
        img: Numpy 3-channel RGB image array to apply canny threshold on
        **kwargs: dictionary containg any of the following keys:
        [color_threshold,abs_x_threshold,abs_y_threshold,
        mag_threshold,dir_threshold,canny_threshold]
        the values must be a dictionary that contains the parameters for each
        thresholding.
    Returns:
        binary_mask: Numpy array binary mask of the thresholded channel
    """
    thresholds = {}
    if 'color_threshold' in kwargs.keys():
        color_threshold_dict = kwargs['color_threshold']
        color_thresh = combine_color_threshold(img, **color_threshold_dict)
        thresholds['color_thresh'] = color_thresh
    if 'abs_x_threshold' in kwargs.keys():
        x_threshold_dict = kwargs['abs_x_threshold']
        x_thresh = abs_sobel_threshold(img, **x_threshold_dict)
        thresholds['x_thresh'] = x_thresh
    if 'abs_y_threshold' in kwargs.keys():
        y_threshold_dict = kwargs['abs_y_threshold']
        y_thresh = abs_sobel_threshold(img, **y_threshold_dict)
        thresholds['y_thresh'] = y_thresh
    if 'mag_threshold' in kwargs.keys():
        mag_threshold_dict = kwargs['mag_threshold']
        mag_thresh = mag_threshold(img, **mag_threshold_dict)
        thresholds['mag_thresh'] = mag_thresh
    if 'dir_threshold' in kwargs.keys():
        mag_threshold_dict = kwargs['mag_threshold']
        dir_thresh = mag_threshold(img, **mag_threshold_dict)
        thresholds['dir_thresh'] = dir_thresh
    if 'canny_threshold' in kwargs.keys():
        canny_threshold_dict = kwargs['canny_threshold']
        canny_thresh = canny_threshold(img, **canny_threshold_dict)
        thresholds['canny_thresh'] = canny_thresh
    if 'laplace_threshold' in kwargs.keys():
        laplace_threshold_dict = kwargs['laplace_threshold']
        laplace_thresh = laplace_threshold(img, **laplace_threshold_dict)
        thresholds['laplace_thresh'] = laplace_thresh
    if 'morph_threshold' in kwargs.keys():
        morph_threshold_dict = kwargs['morph_threshold']
        morph_thresh = morphology_threshold(img, **morph_threshold_dict)
        thresholds['morph_thresh'] = morph_thresh
    return thresholds


def combine_thresholds(thresholds={}, expr=None):
    """Combines the various gradient thresholds.
    Input:
        thresholds: dictionary containing the various thresholds.
        each key: value pair represents the thresholding type and
        a numpy array image mask array
        expr: optional string expression to combine the thresholds.
    Returns:
        combined: combined binay mask of the image array
    """
    combined = np.zeros_like(thresholds['x_thresh'])
    if not expr:
        combined[
            thresholds['color_thresh'] == 1 |
            (thresholds['x_thresh'] == 1) |
            (thresholds['y_thresh'] == 1) |
            (thresholds['mag_thresh'] == 1) |
            (thresholds['dir_thresh'] == 1)] = 1
    else:
        exec(expr)
    return combined


def roi_threshold(img, vertices=None):
    """Applies a static region of interest masking on the input image.
    Args:
        img(ndarray): Numpy input array
    """
    if not vertices:
        vertices = get_mask_vertices(img.shape)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def morphology_threshold(img, kernel=5, thresh=(25, 100)):

    HLS_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    HLS_img = norm(HLS_img[:, :, 2])
    LAB_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    LAB_img = norm(LAB_img[:, :, 2])
    LUV_img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    LUV_img = norm(LUV_img[:, :, 0])
    stacked_img = np.dstack((HLS_img, LAB_img, LUV_img))
    gray_stacked = gray_scale(stacked_img)
    kernel = np.ones((kernel, kernel), np.uint8)
    gradient = cv2.morphologyEx(gray_stacked, cv2.MORPH_GRADIENT, kernel)
    ret, thresh = cv2.threshold(
        gradient, thresh[0], thresh[1], cv2.THRESH_BINARY)
    return thresh
