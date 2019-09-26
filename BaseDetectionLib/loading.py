import cv2 as cv  # Base CV Libraries
import numpy as np  # To easily mess with cv image arrays

# Other stuff
from matplotlib import pyplot as plt  # For plot showing
from scipy import signal  # For peak detection
from collections import deque  # For rolling average

# # -----------------------------------------------------------------------------------------------------
# START OF RELIABLE LANE MARKINGS
# Global for rolling average
rollingLeft = deque(maxlen=20)
rollingRight = deque(maxlen=20)
plt.ion()
fig = None  # plt.figure()


def reliable_lane_markings(image, ipm_points, progress_display=False, ipm_size=200):
    global fig
    if progress_display and fig is None:
        fig = plt.figure()
    """ Applies the reliable_lane_markings process

    :param image: The input image
    :param ipm_points:
    :param progress_display:
    :param ipm_size:
    :return:
    """

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
        rollingLeft.append(peaks[np.argmax(weights[peaks[peaks < ipm_size/2]])])
        rollingRight.append(peaks[np.argmax(weights[peaks[peaks > ipm_size/2]]) + len(peaks[peaks < ipm_size/2])])

    max_left = int(np.average(rollingLeft))  # Get the new average
    max_right = int(np.average(rollingRight))

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
    return image


# -----------------------------------------------------------------------------------------------------
# START OF SDC LANE DETECTION
def sdc_lane_detection(image, roi_rect, apply_roi=True, progress_display=False):
    # Apply the mask to decrease processing later
    if apply_roi:
        mask_image = image[roi_rect[1]:roi_rect[3], roi_rect[0]:roi_rect[2], :]  # Stack is to make image 3 channel
    else:
        mask_image = image  # If no clip, just pass through the full image
    # RGB Threshold
    mask_rgb = cv.inRange(mask_image, (130, 130, 130), (180, 180, 180))

    # HSV Threshold
    mask_hsv = cv.cvtColor(mask_image, cv.COLOR_BGR2HSV)
    mask_hsv = cv.inRange(mask_hsv, (80, 0, 140), (120, 20, 165))

    # abs_sobel = np.abs(cv.Sobel(image_gray, cv.CV_64F, 1, 0))
    # abs_sobel = np.abs(cv.Sobel(gray, cv.CV_64F, 0, 1))
    # # Rescale back to 8 bit integer
    # scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # # Create a copy and apply the threshold
    # binary_output = np.zeros_like(scaled_sobel)
    # # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    # binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Sobel filter
    image_gray = cv.cvtColor(mask_image, cv.COLOR_BGR2GRAY)
    image_gray = cv.equalizeHist(image_gray)
    mask_sobelx = np.abs(cv.Sobel(image_gray, cv.CV_64F, 1, 0, ksize=5))
    mask_sobely = np.abs(cv.Sobel(image_gray, cv.CV_64F, 0, 1, ksize=5))

    # Rescale to 8 bit interger
    mask_sobelx = np.uint8(255*mask_sobelx/np.max(mask_sobelx))
    mask_sobely = np.uint8(255*mask_sobely/np.max(mask_sobely))

    # Laplacian filter
    mask_laplacian = cv.Laplacian(image_gray, cv.CV_32F)

    # And the sobel and Laplacian
    mask_edge_comb = cv.bitwise_or(mask_sobelx, mask_sobely)
    # mask_edge_comb = cv.inRange(mask_edge_comb, 40, 255)
    mask_edge_comb = cv.medianBlur(mask_edge_comb, 5)*2
    mask_edge_comb = cv.bilateralFilter(mask_edge_comb, 5, 10, 100)

    # Clip to the ROI, as it will make some of the binary functions run faster
    full_mask = mask_rgb | mask_hsv

    # mask_rgb = cv.morphologyEx(mask_rgb, cv.MORPH_CLOSE, (3, 3))
    filtered_mask = cv.morphologyEx(full_mask, cv.MORPH_DILATE, (10, 5), iterations=1)
    filtered_mask = cv.medianBlur(filtered_mask, 5)

    if progress_display:
        cv.imshow("SDC: mask_image", mask_image)
        cv.imshow("SDC: mask_rgb", mask_rgb)
        cv.imshow("SDC: mask_hsv", mask_hsv)
        cv.imshow("SDC: mask_sobelx", mask_sobelx)
        cv.imshow("SDC: mask_sobely", mask_sobely)
        cv.imshow("SDC: mask_laplacian", mask_laplacian)
        cv.imshow("SDC: mask_edge_comb", mask_edge_comb)
        cv.imshow("SDC: full_mask", full_mask)
        cv.imshow("SDC: filtered_mask", filtered_mask)
    return image


def neural_lane(image, progress_display=False):
    return image
