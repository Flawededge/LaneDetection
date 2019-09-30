import cv2 as cv  # Base CV Libraries
import numpy as np  # To easily mess with cv image arrays
import math  # Cause maths is fun
from scipy.stats import linregress  # For calculating line

# Other stuff
from matplotlib import pyplot as plt  # For plot showing
from scipy import signal  # For peak detection
from collections import deque  # For rolling average

# # -----------------------------------------------------------------------------------------------------
# START OF RELIABLE LANE MARKINGS
# Global for rolling average
relRollingLeft = deque(maxlen=20)
relRollingRight = deque(maxlen=20)
plt.ion()
fig = None  # plt.figure()


def reliable_lane_markings(image, ipm_points, progress_display=False, passthrough_image = None, ipm_size=200):
    """ Applies the reliable lane markings algorithm

    :param image: The input image. (960x, 540y)
    :type image: np.ndarray
    :param ipm_points: The x, y points for the IPM. Top left, top right, bottom right, bottom left
    :param progress_display:
    :type progress_display: bool
    :param ipm_size: The x*x size of the ipm within the process
    :type ipm_size: int
    :return:
    """

    global fig
    if progress_display and fig is None:
        fig = plt.figure()

    # Make the output perspective points
    ipm_output = np.float32([  # Setup the output transform
        (0, 0),  # Top left
        (ipm_size, 0),  # Top right
        (ipm_size, ipm_size),  # Bottom right
        (0, ipm_size)  # Bottom left
    ])

    # Perspective transform
    m = cv.getPerspectiveTransform(ipm_points, ipm_output)  # Get transform
    perspective = cv.warpPerspective(image, m, (int(ipm_size * 1), int(ipm_size * 1)))  # Apply it
    perspective = cv.bilateralFilter(perspective, 11, 180, 120)

    # Get the lane lines
    gray_perspective = cv.cvtColor(perspective, cv.COLOR_BGR2HSV)[:, :, 2]
    gray_perspective = cv.adaptiveThreshold(gray_perspective, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv.THRESH_BINARY, 9, -1)
    gray_perspective = cv.dilate(gray_perspective, (7, 7))
    gray_perspective = cv.bilateralFilter(gray_perspective, 15, 100, 100)

    thresh_perspective = cv.medianBlur(gray_perspective, 7)
    thresh_perspective = cv.dilate(thresh_perspective, (3, 3))
    thresh_perspective = cv.morphologyEx(thresh_perspective, cv.MORPH_CLOSE, (7, 7), iterations=20)

    # Calculate the weight of each column
    weights = np.array([np.average(i) for i in np.transpose(thresh_perspective)])

    if progress_display:  # For pyplot display
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(weights)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Get the index of the peaks from the graph
    peaks, _ = signal.find_peaks(weights)

    # Append the new average to the average
    if len((weights[peaks[peaks < ipm_size/2]])) and len((weights[peaks[peaks > ipm_size/2]])):
        relRollingLeft.append(peaks[np.argmax(weights[peaks[peaks < ipm_size / 2]])])
        relRollingRight.append(peaks[np.argmax(weights[peaks[peaks > ipm_size / 2]]) + len(peaks[peaks < ipm_size / 2])])

    max_left = int(np.average(relRollingLeft))  # Get the new average
    max_right = int(np.average(relRollingRight))

    # Draw the lines to show
    cv.line(perspective, (max_left, 0), (max_left, int(ipm_size)), (255, 0, 0), 1)
    cv.line(perspective, (max_right, 0), (max_right, int(ipm_size)), (255, 0, 0), 1)

    # # Inverse perspective transform
    # Make the mask to be inverted
    mask_perspective = np.zeros_like(perspective)
    cv.rectangle(mask_perspective, (max_left, 0), (max_right, int(ipm_size)), (0, 0, 255), -1)
    cv.line(mask_perspective, (max_left, 0), (max_left, int(ipm_size)), (255, 0, 0), 5)
    cv.line(mask_perspective, (max_right, 0), (max_right, int(ipm_size)), (255, 0, 0), 5)

    # Do the transform
    m = cv.getPerspectiveTransform(ipm_output, ipm_points)
    output_mask = cv.warpPerspective(mask_perspective, m, (image.shape[1], image.shape[0]))  # Apply it
    image = cv.addWeighted(image, 1, output_mask, 0.5, 0)

    if progress_display:
        cv.imshow("Reliable: Perspective", perspective)
        cv.imshow("Reliable: PerspecGray", gray_perspective)
        cv.imshow("Reliable: PerspThresh", thresh_perspective)
        cv.imshow("Reliable: PerspecMask", mask_perspective)
        cv.imshow("Reliable:  outputMask", output_mask)

    cv.line(image, tuple(ipm_points[0]), tuple(ipm_points[1]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[1]), tuple(ipm_points[2]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[2]), tuple(ipm_points[3]), (191, 66, 245), 2)
    cv.line(image, tuple(ipm_points[3]), tuple(ipm_points[0]), (191, 66, 245), 2)

    if passthrough_image is not None:
        passthrough_image = cv.addWeighted(passthrough_image, 1, output_mask, 0.5, 0)
        return image, passthrough_image

    return image


# -----------------------------------------------------------------------------------------------------
# START OF SDC LANE DETECTION
sdcRollingLeft = deque(maxlen=50)
sdcRollingRight = deque(maxlen=50)


def get_len(points):
    p1 = (points[0][0], points[0][1])
    p2 = (points[0][2], points[0][3])

    dist = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))  # Calculate distance
    rho = math.atan2((p2[1] - p1[1]), (p2[0] - p1[1])) * 57.2958  # Calculate gradient
    return [points, dist, rho]


def extend_points(points, y_coords):  # Takes in x, y, x, y and outputs them extended to extremes
    x2, y2, x1, y1 = points

    slope, intercept, _, _, _ = linregress([x1, x2], [y1, y2])

    points[0] = (y_coords[0]-intercept)/slope  # Line at the top of the screen
    points[1] = y_coords[0]  # Top of the screen has y coordinate 0

    points[2] = (y_coords[1]-intercept)/slope
    points[3] = y_coords[1]

    return points


def sdc_lane_detection(image, roi_rect, line_extrapolate, apply_roi=True, passthrough_image=None, progress_display=False):
    # Apply the mask to decrease processing later
    cv.imwrite("testImage.png", image)
    if apply_roi:
        mask_image = image[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2], :]  # Stack is to make image 3 channel
    else:
        mask_image = image  # If no clip, just pass through the full image

    # RGB Threshold
    mask_rgb = cv.inRange(mask_image, (130, 130, 130), (180, 180, 180))

    mask_rgb = cv.morphologyEx(mask_rgb, cv.MORPH_DILATE, (5, 5), iterations=3)

    # HSV Threshold
    mask_hsv = cv.cvtColor(mask_image, cv.COLOR_BGR2HSV)
    mask_hsv = cv.inRange(mask_hsv, (80, 0, 140), (120, 20, 165))

    mask_hsv = cv.morphologyEx(mask_hsv, cv.MORPH_DILATE, (5, 5), iterations=3)

    # Sobel filter
    image_gray = cv.cvtColor(mask_image, cv.COLOR_BGR2GRAY)
    image_gray = cv.equalizeHist(image_gray)
    mask_sobel_x = np.abs(cv.Sobel(image_gray, cv.CV_64F, 1, 0, ksize=5))
    mask_sobel_y = np.abs(cv.Sobel(image_gray, cv.CV_64F, 0, 1, ksize=5))

    # Rescale to 8 bit integer
    mask_sobel_x = np.uint8(255*mask_sobel_x/np.max(mask_sobel_x))
    mask_sobel_y = np.uint8(255*mask_sobel_y/np.max(mask_sobel_y))

    # Combine and process the final masks
    full_mask = ((mask_rgb | mask_hsv) & (mask_sobel_x + mask_sobel_y))*2
    filtered_mask = cv.morphologyEx(full_mask, cv.MORPH_DILATE, (10, 5), iterations=2)
    filtered_mask = cv.medianBlur(filtered_mask, 5)

    # Have a rolling average to get rid of some flickering and noise
    # if sdcRollingAverage is None:
    #     sdcRollingAverage = filtered_mask
    # else:
    #     sdcRollingAverage = cv.addWeighted(sdcRollingAverage, .25, filtered_mask, 0.75, 1)

    # Create a hough lines display image
    # hough_lines = sdcRollingAverage
    lines = cv.HoughLinesP(filtered_mask, 1, np.pi / 180, 50, None, 50, 10)
    hough_lines = cv.cvtColor(filtered_mask, cv.COLOR_GRAY2BGR)

    # points, dist, rho
    if lines is not None:
        lines = np.array(list(map(get_len, lines)))

        left = np.array([i for i in lines if (-14 > i[2] > -16.8)])
        right = np.array([i for i in lines if (5 < i[2] < 9)])

        if left.shape[0]:
            tmp = left.transpose()[1].argmax()
            sdcRollingLeft.append(left[tmp][0][0])

        if right.shape[0]:
            tmp = right.transpose()[1].argmax()
            sdcRollingRight.append(right[tmp][0][0])

        [cv.line(hough_lines, (i[0][0][0], i[0][0][1]), (i[0][0][2], i[0][0][3]), (189, 235, 52), 1, cv.LINE_AA) for i in left]
        [cv.line(hough_lines, (i[0][0][0], i[0][0][1]), (i[0][0][2], i[0][0][3]), (235, 52, 198), 1, cv.LINE_AA) for i in right]

    lef_lane = np.mean(sdcRollingLeft, axis=0, dtype=np.int)
    rig_lane = np.mean(sdcRollingRight, axis=0, dtype=np.int)
    cv.line(hough_lines, (lef_lane[0], lef_lane[1]), (lef_lane[2], lef_lane[3]), (168, 50, 125), 3, cv.LINE_AA)
    cv.line(hough_lines, (rig_lane[0], rig_lane[1]), (rig_lane[2], rig_lane[3]), (105, 168, 50), 3, cv.LINE_AA)

    lef_lane = extend_points([roi_rect[0], roi_rect[1], roi_rect[0], roi_rect[1]] + lef_lane, line_extrapolate)
    rig_lane = extend_points([roi_rect[0], roi_rect[1], roi_rect[0], roi_rect[1]] + rig_lane, line_extrapolate)

    # Generate the output mask
    output_mask = np.zeros_like(image)
    cv.line(output_mask, (lef_lane[0], lef_lane[1]), (lef_lane[2], lef_lane[3]), (168, 50, 125), 3, cv.LINE_AA)
    cv.line(output_mask, (rig_lane[0], rig_lane[1]), (rig_lane[2], rig_lane[3]), (105, 168, 50), 3, cv.LINE_AA)

    # Apply the mask
    image = cv.addWeighted(image, 1, output_mask, 0.5, 0)

    if progress_display:
        cv.imshow("SDC: mask_image", mask_image)
        cv.imshow("SDC: mask_rgb", mask_rgb)
        cv.imshow("SDC: mask_hsv", mask_hsv)
        cv.imshow("SDC: mask_sobel_x", mask_sobel_x)
        cv.imshow("SDC: mask_sobel_y", mask_sobel_y)
        cv.imshow("SDC: full_mask", full_mask)
        cv.imshow("SDC: filtered_mask", filtered_mask)
        # cv.imshow("SDC: final_averaged", sdcRollingAverage)
        cv.imshow("SDC: hough_lines", hough_lines)

    if passthrough_image is not None:
        passthrough_image = cv.addWeighted(passthrough_image, 1, output_mask, 0.5, 0)
        return image, passthrough_image

    return image


def neural_lane(image, progress_display=False):
    return image
