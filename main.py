import cap as cap
import cv2
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *

# initialization
# calibration = CameraCalibration("scacco.it", 7, 7, True)
calibration = CameraCalibration("chessboard_imgs", 8, 5, True)
thresholding = Thresholding()
transform = PerspectiveTransformation()
lanelines = LaneLines()

# class vars
M_inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
y_bottom = transform.getYBottom()
y_top = transform.getYTop()
x_bot_left = transform.getXBotLeft()
x_bot_right = transform.getXBotRight()
x_top_left = transform.getXTopLeft()
x_top_right = transform.getXTopRight()
XLT = [[x_top_left, y_top, 0]]
XRT = [[x_top_right, y_top, 0]]
XLB = [[x_bot_left, y_bottom, 0]]
XRB = [[x_bot_right, y_bottom, 0]]
bl = (x_bot_left, y_bottom)
br = (x_bot_right, y_bottom)
tl = (x_top_left, y_top)
tr = (x_top_right, y_top)

# read video from file
cap = cv2.VideoCapture('../video//20230317_151040.mp4')
# cap = cv2.VideoCapture('../video//20230317_145749.mp4')
# cap = cv2.VideoCapture('../video//project_video.mp4') # ?
# cap = cv2.VideoCapture('../video//20230303_145525 tagliato.mp4') # histogram clipLimit=3.0
# cap = cv2.VideoCapture('../video//20230303_145747.mp4') # histogram clipLimit=3.0
# cap = cv2.VideoCapture('../video//20230303_150354.mp4') # bad bad
# cap = cv2.VideoCapture('../video//20230303_151659.mp4') # no good
# cap = cv2.VideoCapture('../video//20230303_151722.mp4') # histogram clipLimit=2.0
# cap = cv2.VideoCapture('../video//20230303_152000 tagliato.mp4')
# cap = cv2.VideoCapture('../video//20230317_145721.mp4') # nah
# cap = cv2.VideoCapture('../video//20230317_145830.mp4') # nope
# cap = cv2.VideoCapture('../video//20230317_145931.mp4') # nope
# cap = cv2.VideoCapture('../video//20230317_150400.mp4') # nah
# cap = cv2.VideoCapture('../video//20230317_150849.mp4')
# cap = cv2.VideoCapture('../video//20230317_151002.mp4') # ni
# cap = cv2.VideoCapture('../video//20230317_152054.mp4') # no

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f"Frame Per second: {video_fps} \nTotal Frames: {total_frames} \n Height: {height} \nWidth: {width}")

new_width = width / 2
new_height = height / 2

if not cap.isOpened():
    print("Error opening video file")

count = 0

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        img = cv2.resize(frame, (1280, 720))
        out_img = np.copy(img)

        showROI = False
        computeROIvariations = True

        transform.reBox(XLB, XLT, XRB, XRT)
        transform.computeM()
        M_inv = transform.computeMinv()
        # img = calibration.undistort(img, True)
        img = transform.forward(img)
        img = thresholding.forward(img)
        img = lanelines.forward(img)
        img = transform.backward(img)
        transform.reBox(XLB, XLT, XRB, XRT)

        scale = 80
        if not computeROIvariations:
            out_img = img
        else:
            out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)

            XLT = lanelines.getXLTop()
            XLT = np.float32(np.array([[[XLT, 0]]]))
            # print(XLT)
            XLT = cv2.perspectiveTransform(XLT, M_inv)[0]
            """#print("warp: " + str(XLT))
            center = (int(XLT[0][0]), y_top)
            out_img = cv2.circle(out_img, center, 20, (238, 175, 224), 12)"""

            XLB = lanelines.getXLBot()
            XLB = np.float32(np.array([[[XLB, 720]]]))
            XLB = cv2.perspectiveTransform(XLB, M_inv)[0]
            """#print("warp: " + str(XLB))
            center = (int(XLB[0][0]), y_bottom)
            out_img = cv2.circle(out_img, center, 15, (238, 175, 224), 12)"""

            XRT = lanelines.getXRTop()
            XRT = np.float32(np.array([[[XRT, 0]]]))
            XRT = cv2.perspectiveTransform(XRT, M_inv)[0]
            """#print("warp: " + str(XRT))
            center = (int(XRT[0][0]), y_top)
            out_img = cv2.circle(out_img, center, 5, (238, 175, 224), 12)"""

            XRB = lanelines.getXRBot()
            XRB = np.float32(np.array([[[XRB, 720]]]))
            XRB = cv2.perspectiveTransform(XRB, M_inv)[0]
            """#print("warp: " + str(XRB))
            center = (int(XRB[0][0]), y_bottom)
            out_img = cv2.circle(out_img, center, 10, (238, 175, 224), 12)"""

            if 0 <= XLB[0][0] <= 1280 and 0 <= XLT[0][0] <= 1280 and 0 <= XRB[0][0] <= 1280 and 0 <= XRT[0][0] <= 1280:
                out_img = lanelines.plot(out_img, XLB, XLT, XRB, XRT)

        if showROI:
            x_bot_left = XLB[0][0] - 40 if 0 <= XLB[0][0] - 40 else 0
            x_bot_right = XRB[0][0] + 40 if XRB[0][0] + 40 <= 1280 else 1280
            x_top_left = XLT[0][0] - 20 if 0 <= XLT[0][0] - 20 else 0
            x_top_right = XRT[0][0] + 20 if XRT[0][0] + 20 <= 1280 else 1280

            bl = (int(x_bot_left), int(y_bottom))
            br = (int(x_bot_right), int(y_bottom))
            tl = (int(x_top_left), int(y_top))
            tr = (int(x_top_right), int(y_top))

            cv2.line(out_img, bl, tl, (0, 0, 255), 4)  # bot-lef to top-lef
            cv2.line(out_img, bl, br, (0, 0, 255), 4)  # bot-lef to bot-rig
            cv2.line(out_img, br, tr, (0, 0, 255), 4)  # bot-rig to top-rig
            cv2.line(out_img, tr, tl, (0, 0, 255), 4)  # top-rig to top-lef

        height = int(out_img.shape[0] * scale / 100)
        width = int(out_img.shape[1] * scale / 100)
        dim = (width, height)
        resized = cv2.resize(out_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('img', resized)
        # Press Q on keyboard to exit
        cv2.waitKey(00) == ord('k')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
