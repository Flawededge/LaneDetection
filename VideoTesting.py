# https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking

# Select which file to use
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from IPython.display import clear_output
import processes
from os import _exit

# Define which folder the videos are in and which index to use
path = Path("C:\\Users\\crdig\\Desktop\\Golf-cart-dataset\\My Videos")
number = 0

files = [i for i in path.iterdir()]
[print(f"{cnt}|\t{i}") for cnt, i in enumerate(files)]

targetFile = files[number]
print(f"\nLoading '{targetFile.parts[-1]}' at {targetFile}")

cap = cv.VideoCapture(str(targetFile))  # Get the video capture stream
progress = range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
cv.namedWindow('frame', cv.WINDOW_GUI_EXPANDED)  # Build a named window which can be resized
cv.resizeWindow('frame', 1000, 900)

# while cap.isOpened():
for i in tqdm(progress):
    ret, frame = cap.read()

    frame = processes.process_image(frame)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q') or i > 10:
        break

cap.release()
cv.destroyAllWindows()
_exit(1)
