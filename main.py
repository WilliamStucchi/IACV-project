import cv2
from CameraCalibration import CameraCalibration

# initialization
calibration = CameraCalibration("scacco.it", 7, 7)

print("END")

# read video from file
cap = cv2.VideoCapture('../video//20230303_145434.mp4')

video_fps = cap.get(cv2.CAP_PROP_FPS),
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

        # all here
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # temp = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        # temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        L, A, B = cv2.split(temp)
        # cv2.imshow('Frame', temp)
        cv2.imshow('L_Channel', L)
        cv2.imshow('A_Channel', A)
        cv2.imshow('B_Channel', B)




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
