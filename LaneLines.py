import cv2
import numpy as np
import matplotlib.image as mpimg


def hist(img):
    bottom_half = img[img.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


class LaneLines:
    """ Class containing information about detected lane lines.
    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """

    car_offset = 0
    XLT = 0
    XRT = 0
    XLB = 0
    XRB = 0

    def setCarOffset(self, offset):
        self.car_offset = offset * 700 / 3
        #print(self.car_offset)

    def getCarOffset(self):
        return int(self.car_offset)

    def setXLXRTop(self, xl, xr):
        self.XLT = xl
        self.XRT = xr

    def getXLTop(self):
        return int(self.XLT)

    def getXRTop(self):
        return int(self.XRT)

    def setXLXRBot(self, xl, xr):
        self.XLB = xl
        self.XRB = xr

    def getXLBot(self):
        return int(self.XLB)

    def getXRBot(self):
        return int(self.XRB)

    def __init__(self):
        """Init Lanelines.
        Parameters:
            left_fit (np.array): Coefficients of polynomial that fit left lane
            right_fit (np.array): Coefficients of polynomial that fit right lane
            binary (np.array): binary image
        """
        self.left_fit = None
        self.right_fit = None
        self.binary = None
        self.nonzero = None
        self.nonzerox = None
        self.nonzeroy = None
        self.clear_visibility = True
        self.dir = []
        self.left_curve_img = mpimg.imread('left_turn.png')
        self.right_curve_img = mpimg.imread('right_turn.png')
        self.straight_img = mpimg.imread('straight.png')
        self.left_curve_img = cv2.normalize(src=self.left_curve_img, dst=None, alpha=0, beta=20,
                                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.right_curve_img = cv2.normalize(src=self.right_curve_img, dst=None, alpha=0, beta=20,
                                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.straight_img = cv2.normalize(src=self.straight_img, dst=None, alpha=0, beta=20,
                                               norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # HYPERPARAMETERS
        # Number of sliding windows
        self.nwindows = 9
        # Width of the the windows +/- margin
        self.margin = 100
        # Mininum number of pixels found to recenter window
        self.minpix = 50

    def forward(self, img):
        """Take a image and detect lane lines.
        Parameters:
            img (np.array): An binary image containing relevant pixels
        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height):
        """ Return all pixel that are in a specific window
        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window
        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def extract_features(self, img):
        """ Extract features from a binary image
        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        # Height of windows - based on nwindows and image shape
        self.window_height = int(img.shape[0] // self.nwindows)

        # Identify the x and y positions of all nonzero pixel in the image
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.
        Parameters:
            img (np.array): A binary warped image
        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A RGB image that use to display result later on.
        """
        assert (len(img.shape) == 2)

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((img, img, img))

        # select the starting point of the lines
        histogram = hist(img)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Current position to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height // 2

        # Create empty lists to receive left and right lane pixel
        leftx, lefty, rightx, righty = [], [], [], []

        # Step through the windows one by one
        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height)

            # Append these indices to the lists
            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.
        Parameters:
            img (np.array): a binary warped image
        Returns:
            out_img (np.array): an RGB image that have lane line drawn on that.
        """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])

        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Visualization
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)

            cv2.line(out_img, (l, y), (r, y), (255, 0, 0))

        self.setXLXRTop(left_fitx[0], right_fitx[0])
        self.setXLXRBot(left_fitx[len(left_fitx) - 1], right_fitx[len(right_fitx) - 1])

        lR, rR, pos = self.measure_curvature()

        return out_img

    def plot(self, out_img,  XLB, XLT, XRB, XRT):
        np.set_printoptions(precision=6, suppress=True)
        lR, rR, pos = self.measure_curvature()

        value = None
        if abs(self.left_fit[0]) > abs(self.right_fit[0]):
            value = self.left_fit[0]
        else:
            value = self.right_fit[0]

        if abs(value) <= 0.00015:
            self.dir.append('F')
        elif value < 0:
            self.dir.append('L')
        else:
            self.dir.append('R')

        if len(self.dir) > 10:
            self.dir.pop(0)

        W = 410
        H = 80
        widget = np.copy(out_img[:H, :W])
        widget //= 2
        out_img[:H, :W] = widget

        y_bottom = XLB[0][1]
        y_top = XLT[0][1]
        XL = abs((XLT[0][0] - XLB[0][0]) // 2) + XLB[0][0]
        XR = XRB[0][0] - abs((XRT[0][0] - XRB[0][0]) // 2)
        TB = (y_bottom - y_top) // 2 + y_top # y coord of the beginning of the arrow img
        LB = (XR - XL) // 2 + XL # x coord of the beginning of the arrow img

        direction = max(set(self.dir), key=self.dir.count)
        msg = "Keep Straight Ahead"
        if direction == 'L':
            y, x = self.left_curve_img[:, :, 2].nonzero()
            v_shift = self.left_curve_img.shape[0] // 2  # vertical shift to center the img
            h_shift = self.left_curve_img.shape[1] // 2  # horizontal shift to center the img
            out_img[y + int(TB - v_shift), x + int(LB - h_shift)] = self.left_curve_img[y, x, :3]
            msg = "Left Curve Ahead"

        if direction == 'R':
            y, x = self.right_curve_img[:, :, 2].nonzero()
            v_shift = self.right_curve_img.shape[0] // 2  # horizontal shift to center the img
            h_shift = self.right_curve_img.shape[1] // 2  # horizontal shift to center the img
            out_img[y + int(TB - v_shift), x + int(LB - h_shift)] = self.right_curve_img[y, x, :3]
            msg = "Right Curve Ahead"

        if direction == 'F':
            y, x = self.straight_img[:, :, 2].nonzero()
            v_shift = self.straight_img.shape[0] // 2  # horizontal shift to center the img
            h_shift = self.straight_img.shape[1] // 2  # horizontal shift to center the img
            out_img[y + int(TB - v_shift), x + int(LB - h_shift)] = self.straight_img[y, x, :3]

        cv2.putText(out_img, msg, org=(10, 60), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.6, color=(255, 255, 255),
                    thickness=1)

        cv2.putText(
            out_img,
            "Vehicle is {:.2f} m away from center".format(pos),
            org=(10, 25),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.6,
            color=(255, 255, 255),
            thickness=1)

        return out_img

    def measure_curvature(self):
        ym = 30 / 720
        xm = 3 / 700 # 3m width of the road lane

        left_fit = self.left_fit.copy()
        right_fit = self.right_fit.copy()
        y_eval = 700 * ym

        # Compute R_curve (radius of curvature)
        left_curveR = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        right_curveR = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        xl = np.dot(self.left_fit, [700 ** 2, 700, 1])
        xr = np.dot(self.right_fit, [700 ** 2, 700, 1])
        car_position = 1280 // 2
        lane_centre_position = (xl + xr) // 2
        pos = (car_position - lane_centre_position) * xm
        self.setCarOffset(pos)

        return left_curveR, right_curveR, pos