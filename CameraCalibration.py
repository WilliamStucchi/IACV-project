import numpy as np
import cv2
import glob
import matplotlib.image as mpimg

class CameraCalibration():

    def __init__(self, image_dir, nx, ny, debug):
        """ Init CameraCalibration.
        Parameters:
            image_dir (str): path to folder contains chessboard images
            nx (int): width of chessboard (number of squares)
            ny (int): height of chessboard (number of squares)
        """
        if not debug:
            fnames = glob.glob("{}/*".format(image_dir))
            objpoints = []
            imgpoints = []

            # Coordinates of chessboard's corners in 3D
            objp = np.zeros((nx * ny, 3), np.float32)
            objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

            # Go through all chessboard images
            for f in fnames:
                img = mpimg.imread(f)

                # Convert to grayscale image
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # Find chessboard corners
                ret, corners = cv2.findChessboardCorners(img, (nx, ny))
                if ret:
                    imgpoints.append(corners)
                    objpoints.append(objp)



            shape = (img.shape[1], img.shape[0])
            ret, self.mtx, self.dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
            print(self.mtx)
            print(self.dist)
            if not ret:
                raise Exception("Unable to calibrate camera")

    def undistort(self, img, debug):
        # Convert to grayscale image

        if not debug:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        """
        #scacco.it
        mtx = np.matrix([[1980.87395, 0, 1292.38594],
               [0, 1975.79294, 719.506143],
               [0,          0,          1]])
        dist = np.array([0.442880152, -3.00957728, 0.00301437425, 0.000303952355, 6.71664977])
        """
        #chessboard
        mtx = np.matrix([[2060.48005, 0, 1237.43623],
                         [0, 2053.87150, 695.490058],
                         [0, 0, 1]])
        dist = np.array([0.412293769, -3.19040114, -0.00149741508, -0.013392118, 6.78623286])
        return cv2.undistort(img, mtx, dist, None, mtx)


    """
        chessboardSize = (7, 7)
        frameSize = (2560, 1440)
    
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
    
        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
    
        images = glob.glob('scacco.it/*.jpg')
        for fname in images:
            print(fname)
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
                scale = 50
                height = int(img.shape[0] * scale / 100)
                width = int(img.shape[1] * scale / 100)
                dim = (width, height)
                resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
                cv.imshow('img', resized)
                cv.waitKey(1000)
    
        cv.destroyAllWindows()
    
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    
        print("Camera Calibrated: ",  ret)
        print("\nCamera Matrix:\n", mtx)
        print("\nDistortion Parameters: \n", dist)
        print("\nRotation Vectors: \n", rvecs)
        print("\nTranslation Vectors: \n", tvecs)
    
        img = cv.imread('scacco.it/20230307_134211.jpg')
        h,  w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
        # undistort
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y: y + h, x: x + w]
        cv.imwrite('calibresult.png', dst)
    
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        print("total error: {}".format(mean_error/len(objpoints)))
    """