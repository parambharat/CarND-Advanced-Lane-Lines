"""Utilities to warp and tansform an Image."""
import cv2
import numpy as np


def get_perspective_dst(imshape):
    """Comments."""
    rows, cols = imshape[0], imshape[1]
    dst = np.float32([
        [cols * 0.25, rows * 0.025],
        [cols * 0.25, rows * 0.975],
        [cols * 0.75, rows * 0.025],
        [cols * 0.75, rows * 0.975],
    ])
    return dst


def get_perspective_src(imshape):
    rows, cols = imshape[0], imshape[1]
    src = np.float32([
        [cols * 0.4475, rows * 0.65],
        [cols * 0.175, rows * 0.95],
        [cols * 0.5525, rows * 0.65],
        [cols * 0.825, rows * 0.95],
    ])
    return src


def calculate_intercept(lane_lines, slopes):
    """Comments."""
    slopes = slopes[~np.isnan(slopes)]
    slopes = slopes[~np.isinf(slopes)]
    avg_slope = slopes.mean()
    lane_lines = lane_lines.reshape(
        (lane_lines.shape[0] * 2, lane_lines.shape[1] // 2))
    x_avg, y_avg = np.mean(lane_lines, axis=0)
    return y_avg - (x_avg * avg_slope), avg_slope


def extrapolate_lines(y_1, y_2, slope, intercept):
    """Comments."""

    if not (~np.isnan(slope) and ~np.isnan(intercept)):
        x_1 = y_1 = x_2 = y_2 = 0.0
    else:
        x_1 = (y_1 - intercept) / slope
        x_2 = (y_2 - intercept) / slope

    return np.array([[x_1, y_1], [x_2, y_2]], dtype=np.float32)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Comments."""
    imshape = img.shape
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
        []), minLineLength=min_line_len, maxLineGap=max_line_gap)
    lines = lines.reshape((lines.shape[0], lines.shape[2]))
    y_min = lines.reshape(
        (lines.shape[0] * 2, lines.shape[1] // 2))[:, 1].min()
    slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])
    slopes = slopes[~np.isinf(slopes)]
    slopes = slopes[~np.isnan(slopes)]

    left_lines, right_lines = (lines[slopes < -0.5], lines[slopes > 0.5])
    left_slopes, right_slopes = (slopes[slopes < -0.5], slopes[slopes > 0.5])

    c_left, left_avg_slope = calculate_intercept(left_lines, left_slopes)
    c_right, right_avg_slope = calculate_intercept(right_lines, right_slopes)

    left_lines = extrapolate_lines(y_1=y_min, y_2=imshape[0],
                                   slope=left_avg_slope, intercept=c_left)
    right_lines = extrapolate_lines(y_1=y_min, y_2=imshape[0],
                                    slope=right_avg_slope, intercept=c_right)
    return np.vstack((left_lines, right_lines))


def draw_lines(img, lines, color=[255, 0, 0], thickness=12):
    """Comments."""
    [x1, y1], [x2, y2] = lines[0, :], lines[1, :]
    [x3, y3], [x4, y4] = lines[2, :], lines[3, :]
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    cv2.line(img, (x3, y3), (x4, y4), color, thickness)


def transform_perspective(img, src, dst, reverse=False):
    """Comments."""
    imshape = img.shape
    rows, cols = imshape[0], imshape[1]
    if not dst:
        dst = get_perspective_dst(imshape)
    if not np.all(src) or not (src > 0).all():
        src = get_perspective_src(imshape)

    if (reverse):
        src, dst = dst, src
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (cols, rows))
    return warped, Minv
