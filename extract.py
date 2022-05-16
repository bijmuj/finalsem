import os

import cv2

if not os.path.isdir("video"):
    os.mkdir("video")
video = cv2.VideoCapture("downscaled_trim.mp4")

length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

count = 0
while count < 20:
    is_read, frame = video.read()
    count += 1
    cv2.imwrite(f"video/0_{count}.jpg", frame)

video.release()
