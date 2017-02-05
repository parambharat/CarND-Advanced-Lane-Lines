import os
import numpy as np
import cv2
from .general_utils import gray_scale, save_load


def find_corners(imgs, dims=(9, 6)):
    """
    Finds chessboard corners in a set of images.
    Needed for caliberating the camera
    Inputs:
        imgs: Numpy array with multiple image arrays.
        dims: Dimensions of the chessboard in the image
    Returns:
        tuple: (img_pts, obj_pts)
            img_pts: image points needed for caliberation
            obj_pts: object points needed for caliberation
    """
    nx, ny = dims
    obj_p = np.zeros((nx*ny, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    img_pts, obj_pts = [], []
    for img in imgs:
        gray = gray_scale(img)
        ret, corners = cv2.findChessboardCorners(
            gray, (nx, ny), None)
        if ret:
            img_pts.append(corners)
            obj_pts.append(obj_p)
    return img_pts, obj_pts


def caliberate_camera(img, obj_pts, img_pts):
    """
    Caliberates a camera and computes the distortion coefficients.
    Input:
        img: sample_img needed to compute shape of the images
        obj_pts:  object points needed for caliberation
        img_pts: image points needed for caliberation
    Returns:
        mtx: camera matrix
        dist: distortion coefficients
    """
    imshape = img.shape[0:2]
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, imshape, None, None)
    return mtx, dist


def maybe_load_coeffs(coeff_file, imgs=None):
    """
    Checks if the coeff_file exists

    if it does:
    loads the camera matrix and distortion coefficients and
    returns the camera matrix and distortion coefficients
    Warning: The first run of this function might take a while.

    Else, computes the coeffecients, creates the file and
    returns the camera matrix and distortion coefficients
    Inputs:
        imgs: Numpy array containing caliberation image arrays.
        coeff_file: Absolute path of the coeff file
    Returns:
        mtx: camera matrix
        dist: distortion coefficients
    """
    if not os.path.isfile(coeff_file):
        img_pts, obj_pts = find_corners(imgs, (9, 6))
        mtx, dist = caliberate_camera(
            imgs[0], obj_pts, img_pts)
        obj = {'mtx': mtx, 'dist': dist}
        save_load(coeff_file, obj)
    else:
        coeff = save_load(coeff_file)
        mtx, dist = coeff['mtx'], coeff['dist']
    return mtx, dist


def undistort_img(img, mtx, dist):
    """
    Corrects camera distortion in an image.
    Inputs:
        img: Numpy image array.
        mtx: Camera matrix needed for distortion correction
        dist: Distortionb coefficients needed for distortion correction.
    Returns:
        undist: Undistorted numpy image array
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
