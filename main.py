import cv2
from CameraCalibration import CameraCalibration
from CameraCalibration import CameraCalibration
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)

    # checking for right mouse clicks
    if event == cv2.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)

# initialization
calibration = CameraCalibration("scacco.it", 7, 7, True)
#calibration = CameraCalibration("chessboard_imgs", 8, 5)
thresholding = Thresholding()
transform = PerspectiveTransformation()
lanelines = LaneLines()

# read video from file
cap = cv2.VideoCapture('../video//20230317_151040.mp4')

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
print(f"Frame Per second: {video_fps } \nTotal Frames: {total_frames} \n Height: {height} \nWidth: {width}")

new_width = width / 2
new_height = height / 2

if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        img = cv2.resize(frame, (1280, 720))
        out_img = np.copy(img)

        debug = False
        final = True
        if debug:
            cv2.line(img, (0, 720), (400, 520), (0, 0, 255), 8) # bot-lef to top-lef
            cv2.line(img, (0, 720), (1280, 720), (0, 0, 255), 8) # bot-lef to bot-rig
            cv2.line(img, (1280, 720), (880, 520), (0, 0, 255), 8) # bot-rig to top-rig
            cv2.line(img, (880, 520), (400, 520), (0, 0, 255), 8) # top-rig to top-lef


        img = calibration.undistort(img, True)
        img = transform.forward(img)
        img = thresholding.forward(img)
        img = lanelines.forward(img)
        img = transform.backward(img)

        scale = 80
        if not final:
            height = int(img.shape[0] * scale / 100)
            width = int(img.shape[1] * scale / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        else:
            out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
            out_img = lanelines.plot(out_img)
            height = int(out_img.shape[0] * scale / 100)
            width = int(out_img.shape[1] * scale / 100)
            dim = (width, height)
            resized = cv2.resize(out_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('img', resized)
        cv2.setMouseCallback('img', click_event)
        # Press Q on keyboard to exit
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

