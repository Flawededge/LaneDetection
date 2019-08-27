# https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking

# Select which file to use
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
from IPython.display import clear_output
import processes

# Define which folder the videos are in and which index to use
path = Path("G:\Golf cart dataset\My Videos")
number = 0

files = [i for i in path.iterdir()]
[print(f"{cnt}|\t{i}") for cnt, i in enumerate(files)]

targetFile = files[number]
print(f"\nLoading '{targetFile.parts[-1]}' at {targetFile}")

cap = cv.VideoCapture(str(targetFile))  # Get the video capture stream
progress = range(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
cv.namedWindow('frame', cv.WINDOW_GUI_EXPANDED)  # Build a named window which can be resized
cv.namedWindow('binaryPipeline', cv.WINDOW_GUI_EXPANDED)  # Build a named window which can be resized
cv.waitKey(0)
# while cap.isOpened():
for i in tqdm(progress):
    ret, frame = cap.read()

    binaryProcess = processes.binary_pipeline(frame, True)
    cv.imshow('binaryPipeline', binaryProcess)
    cv.imshow('frame', frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        break

    clear_output(wait=True)

cap.release()
cv.destroyAllWindows()
