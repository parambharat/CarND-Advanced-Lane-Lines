from __future__ import print_function
from six.moves import cPickle as pickle
import math
import cv2
import numpy as np
import matplotlib.pylab as plt
from matplotlib import gridspec


def read_image(impath):
    """
    Reads single image from file.
    Inputs:
        impath: Image Absolute path to the image file.
    Returns:
        3-channel RGB image in numpy uint-8 array
    """
    img = cv2.imread(impath)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gray_scale(img):
    """
    Converts a 3 channel RGB image to 1 channel Grayscale image.
    Inputs:
        img: img, numpy 3-channel image array.
    Returns:
        1-channel gray scale image in numpy uint-8 array
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def norm(image):
    """
    Normalizes an image image.
    Inputs:
        img: img, numpy image array.
    Returns:
        normalized image in numpy uint-8 array
    """
    image = image.astype(np.float32)
    image *= 255.0 / image.max()
    return image.astype(np.uint8)


def do_plot(ax, img, title, cmap=None):
    """
    Plots a single image in an axes
    Inputs:
        ax: matplotlib axes object to plot the image.
        img: Numpy image array to plot.
        title: String title of the image to plot.
        cmap: String colormap to use in plot, defaults to 'jet'
    Returns:
        None, adds image to axis inplace
    """
    if not cmap:
        ax.imshow(img)
        ax.set_title(title, fontsize=30)
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=30)


def plot_images(**kwargs):
    """
    Plots images in a grid of 4 columns.
    Inputs:
        kwargs: Dictionary that must contain the following:
        {'<Image Title>':{
        'image': <image array>
        'cmap': <Color map>, must mention 'gray' for 1-channel images
        }
    Returns:
        None, plot is displayed inline or in qt.
    """
    num_imgs = len(kwargs)
    max_cols = 4
    cols = max_cols if num_imgs / max_cols > 1 else num_imgs
    rows = int(math.ceil(num_imgs / cols))
    rows = rows if rows else 1
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(24, 9))
    for i, (title, img) in enumerate(kwargs.items()):
        ax = fig.add_subplot(gs[i])
        ax.set_axis_off()
        do_plot(ax, img['image'], title, img['cmap'])
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    fig.tight_layout()
    plt.show()


def save_load(f_name, obj={}):
    """
    Save or loads a pickle object to a file.
    Save mode - When both f_name and obj are passed.
    Load mode - When only f_name is passed
    Inputs:
        f_name: Name of file to save or load from.
        obj: Optional, python object to be pickled in save mode.
    Returns:
        None - In save mode
        obj(pickled python object) - In load mode

    """
    if obj and f_name:
        with open(f_name, 'wb') as o_file:
            pickle.dump(obj, o_file)
        print('Co-efficients saved at {}'.format(f_name))
    else:
        print('loading distortion coefficients from {}'.format(f_name))
        with open(f_name, 'rb') as i_file:
            obj = pickle.load(i_file)
        return obj
