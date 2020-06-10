import cv2
import numpy as np

from . import gui


def estimate_homography(original_image, original_points, save_warped_frame=True):

    original_points = np.array([[[145, 376]],
                                [[468, 439]],
                                [[697, 203]],
                                [[477, 160]]])

    x_distance = np.linalg.norm(original_points[0][0] - original_points[1][0]) / original_image.shape[1]
    y_distance = np.linalg.norm(original_points[1][0] - original_points[2][0]) / original_image.shape[0]

    XY1 = 650
    XY2 = 850

    xmin = int(XY1 * x_distance)
    ymin = int(XY1 * y_distance)
    xmax = int(XY2 * x_distance)
    ymax = int(XY2 * y_distance)

    bird_view_points = np.array([[[xmin, ymax]],
                                 [[xmax, ymax]],
                                 [[xmax, ymin]],
                                 [[xmin, ymin]]])

    homography = cv2.findHomography(original_points, bird_view_points)[0]

    if save_warped_frame:
        warped_image = original_image.copy()
        warped_image = cv2.warpPerspective(warped_image, homography, (warped_image.shape[1], warped_image.shape[0]))
        cv2.imwrite('Homography Estimate.jpg', cv2.hconcat([original_image, warped_image]))

    return homography


def get_homography(input_video_path, scaling_factor):

    # Capture the first frame of the video
    video_capture = cv2.VideoCapture(input_video_path)
    _, frame = video_capture.read()
    frame = cv2.resize(frame, dsize=(0,0), fx=scaling_factor, fy=scaling_factor)
    video_capture.release()

    # Launch the GUI and project the frame onto bird's eye view plane
    #points = gui.interactive_gui(frame)
    points = []
    homography = estimate_homography(frame, points)

    # Get distance threshold from GUI
    #distance_threshold, tdp1, tdp2 = get_distance_parameters(points, homography)
    distance_threshold = 0

    return homography, distance_threshold


def transform_coords(orig_coords, homography):
    '''
    Project the original coordinates from the image into the birds eye view plane
    '''
    orig_coords = orig_coords.reshape(orig_coords.shape[0], 1, orig_coords.shape[1])
    new_coords = cv2.perspectiveTransform(orig_coords, homography)[0][0]
    return new_coords


def get_distance_parameters(points, H):

    distance_points = np.array(points[4:6], dtype=np.float32)

    point1 = distance_points[0][np.newaxis, :]
    point2 = distance_points[1][np.newaxis, :]

    transformed_point1 = transform_coords(point1, H)
    transformed_point2 = transform_coords(point2, H)

    distance = np.linalg.norm(transformed_point1 - transformed_point2)

    return distance, transformed_point1, transformed_point2
