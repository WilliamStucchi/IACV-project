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

    clip_limit = 1.0
    r_min_th = 0.6
    r_max_th = 1.0
    l_min_th = 0.6
    l_max_th = 1.0

    def __init__(self):
        """ Init Thresholding."""
        self.init = True
        pass

    def forward(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

        l_channel = hls[:, :, 1]

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        right_lane = threshold_rel(l_channel, self.r_min_th, self.r_max_th)
        left_lane = threshold_rel(l_channel, self.l_min_th, self.l_max_th)

        right_lane[:, :690] = 0
        left_lane[:, 590:] = 0

        img2 = left_lane | right_lane

        return img2
