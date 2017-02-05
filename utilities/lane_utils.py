import numpy as np
import matplotlib.pylab as plt
import cv2


def find_peaks(hist):
    right, left = np.array_split(hist, 2)
    return np.argmax(right), np.argmax(left) + len(right)


def fit_polynomial(left_lane, right_lane):
    left_fit = np.polyfit(left_lane[:, 0], left_lane[:, 1], 2)
    right_fit = np.polyfit(right_lane[:, 0], right_lane[:, 1], 2)
    left_func, right_func = np.poly1d(left_fit), np.poly1d(right_fit)
    return left_func, right_func


def find_initial_lanes(img, n_bands=9, w_width=0.2, tol=50):
    # setup variables
    imshape = img.shape
    height, width = imshape[0], imshape[1]
    band_height = int(height / n_bands)
    margin = int(w_width * width * 0.5)

    hist = np.sum(img[int(imshape[0] / 2):, :], axis=0)
    left_base, right_base = find_peaks(hist)

    left_current, right_current = left_base, right_base

    # Good lists
    glx, gly, grx, gry = [], [], [], []

    for band_n in range(n_bands, 0, -1):
        band_highy, band_lowy = (band_n - 1) * \
            band_height, (band_n) * band_height
        left_lowx, left_highx = (
            left_current - margin, left_current + margin)
        right_lowx, right_highx = (
            right_current - margin, right_current + margin)

        # focus only on the bands
        b_left = img[band_highy:band_lowy, left_lowx:left_highx]
        b_right = img[band_highy:band_lowy, right_lowx:right_highx]

        # get non zero indices from each window
        left_y, left_x = np.nonzero(b_left)
        right_y, right_x = np.nonzero(b_right)

        gly.extend(left_y + band_highy)
        glx.extend(left_x + left_lowx)

        gry.extend(right_y + band_highy)
        grx.extend(right_x + right_lowx)

        if len(glx) > tol:
            left_current = np.int(np.mean(glx))
        if len(grx) > tol:
            right_current = np.int(np.mean(grx))

    left_lane = np.array([gly, glx]).T
    right_lane = np.array([gry, grx]).T
    return left_lane, right_lane


def find_later_lanes(img, w_width, left_fit, right_fit):
    imshape = img.shape
    width = imshape[1]
    margin = int(w_width * width * 0.5)

    nzy, nzx = img.nonzero()
    left_lane_inds = (
        (
            nzx > (
                left_fit[0] * (nzy**2) +
                left_fit[1] * nzy +
                left_fit[2] - margin)
        ) &
        (
            nzx < (
                left_fit[0] * (nzy**2) +
                left_fit[1] * nzy +
                left_fit[2] + margin)
        )
    )
    right_lane_inds = (
        (
            nzx > (
                right_fit[0] * (nzy**2) +
                right_fit[1] * nzy +
                right_fit[2] - margin)
        ) &
        (
            nzx < (
                right_fit[0] * (nzy**2) +
                right_fit[1] * nzy +
                right_fit[2] + margin)
        )
    )

    # Again, extract left and right line pixel positions
    leftx, lefty = nzx[left_lane_inds], nzy[left_lane_inds]
    rightx, righty = nzx[right_lane_inds], nzy[right_lane_inds]

    left_lane = np.array([lefty, leftx])
    right_lane = np.array([righty, rightx])
    return left_lane, right_lane


def draw_lanes(img, left_fit, right_fit):
    # Generate x and y values for plotting
    imshape = img.shape
    out_img = np.dstack([img] * 3)

    fity = np.linspace(0, imshape[0] - 1, imshape[0])
    fit_leftx = left_fit(fity)
    fit_rightx = right_fit(fity)
    plt.imshow(out_img)
    plt.plot(fit_leftx, fity, color='yellow')
    plt.plot(fit_rightx, fity, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()


def measure_roc(poly, y_val):
    first_d = poly.deriv(1)
    sec_d = poly.deriv(2)
    first_d, sec_d = first_d(y_val), sec_d(y_val)
    r_curv = ((1 + first_d**2)**(1.5)) / np.absolute(sec_d)
    return r_curv


def find_roc(img, left_poly, right_poly):
    imshape = img.shape
    fity = np.linspace(0, imshape[0] - 1, imshape[0])
    left_curv = measure_roc(left_poly, np.min(fity))
    right_curv = measure_roc(right_poly, np.min(fity))
    print(left_curv, right_curv)


def find_real_curv(img, left_lane, right_lane):
    imshape = img.shape
    y_conv, x_conv = 30 / 720, 3.7 / 700
    imshape = img.shape
    fity = np.linspace(0, imshape[0] - 1, imshape[0])
    left_fit = np.polyfit(
        left_lane[:, 0] * y_conv, left_lane[:, 1] * x_conv, 2)
    right_fit = np.polyfit(
        right_lane[:, 0] * y_conv, right_lane[:, 1] * x_conv, 2)
    left_func, right_func = np.poly1d(left_fit), np.poly1d(right_fit)

    left_curv = measure_roc(left_func, np.min(fity * y_conv))
    right_curv = measure_roc(right_func, np.min(fity * y_conv))

    return left_curv, right_curv


def paint_road(warped_img, undistorted_sample, left_fit, right_fit, Minv):
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    imshape = warped_img.shape
    yvals = np.linspace(0, imshape[0] - 1, imshape[0])
    left_fitx, right_fitx = left_fit(yvals), right_fit(yvals)
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (imshape[1], imshape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_sample, 1, newwarp, 0.3, 0)
    return result
