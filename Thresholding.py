import cv2
import numpy as np


def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)

    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    np.uint8((img >= vlo) & (img <= vhi))
    return np.uint8((img >= vlo) & (img <= vhi)) * 255


def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255


class Thresholding:
    """ This class is for extracting relevant pixels in an image.
    """

    init = True
    non_zero_l = 0
    non_zero_v = 0

    def __init__(self):
        """ Init Thresholding."""
        self.init = True
        pass

    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:, :, 1]

        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        right_lane = threshold_rel(l_channel, 0.6, 1.0)
        left_lane = threshold_rel(l_channel, 0.6, 1.0)

        right_lane[:, :690] = 0
        left_lane[:, 590:] = 0

        img2 = left_lane | right_lane

        return img2
