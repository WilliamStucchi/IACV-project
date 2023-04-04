import cv2
import numpy as np


class PerspectiveTransformation:
    """ This a class for transforming image between front view and top view
    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array): Matrix to transform image from front view to top view
        M_inv (np.array): Matrix to transform image from top view to front view
    """

    y_bottom = 720
    y_top = 500

    ORIGINAL_XBL = 80
    ORIGINAL_XBR = 1180
    ORIGINAL_XTL = 480
    ORIGINAL_XTR = 850

    x_bot_left = ORIGINAL_XBL
    x_bot_right = ORIGINAL_XBR
    x_top_left = ORIGINAL_XTL
    x_top_right = ORIGINAL_XTR

    def __init__(self):
        """Init PerspectiveTransformation."""
        self.src = np.float32([(self.x_top_left, self.y_top),  # top-left
                               (self.x_bot_left, self.y_bottom),  # bottom-left
                               (self.x_bot_right, self.y_bottom),  # bottom-right
                               (self.x_top_right, self.y_top)])  # top-right

        self.dst = np.float32([(0, 0),
                               (0, 720),
                               (1280, 720),
                               (1280, 0)])
        self.computeM()
        self.computeMinv()

    def getYBottom(self):
        return self.y_bottom

    def getYTop(self):
        return self.y_top

    def getXBotLeft(self):
        return self.x_bot_left

    def getXBotRight(self):
        return self.x_bot_right

    def getXTopLeft(self):
        return self.x_top_left

    def getXTopRight(self):
        return self.x_top_right

    def reBox(self, XLB, XLT, XRB, XRT):  # if we want to consider the lines positions
        self.x_bot_left = XLB[0][0] - 40 if 0 <= XLB[0][0] - 40 else 0
        self.x_bot_right = XRB[0][0] + 40 if XRB[0][0] + 40 <= 1280 else 1280
        self.x_top_left = XLT[0][0] - 20 if 0 <= XLT[0][0] - 20 else 0
        self.x_top_right = XRT[0][0] + 20 if XRT[0][0] + 20 <= 1280 else 1280

        if self.x_bot_right - self.x_bot_left <= self.x_top_right - self.x_top_left \
                or self.x_top_right - self.x_top_left <= 200:

            self.x_top_left = self.ORIGINAL_XTL
            self.x_top_right = self.ORIGINAL_XTR
            self.x_bot_left = self.ORIGINAL_XBL
            self.x_bot_right = self.ORIGINAL_XBR

        self.src = np.float32([(self.x_top_left, self.y_top),  # top-left
                               (self.x_bot_left, self.y_bottom),  # bottom-left
                               (self.x_bot_right, self.y_bottom),  # bottom-right
                               (self.x_top_right, self.y_top)])  # top-right

    def computeM(self):
        #print(self.src)
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)

    def computeMinv(self):
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)
        return self.M_inv

    def forward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view
        Parameters:
            img (np.array): A front view image
            img_size (tuple): Size of the image (width, height)
            flags : flag to use in cv2.warpPerspective()
        Returns:
            Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
        """ Take a top view image and transform it to front view
        Parameters:
            img (np.array): A top view image
            img_size (tuple): Size of the image (width, height)
            flags (int): flag to use in cv2.warpPerspective()
        Returns:
            Image (np.array): Front view image
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)

    def pointBackTransform(self, point, dim = (1, 1), flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(point, self.M_inv, dim, flags=flags)

