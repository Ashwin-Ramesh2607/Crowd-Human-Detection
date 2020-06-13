import copy
import time
import argparse

import cv2
import numpy as np

from utils import plots
from utils import visualize
from utils import get_params
from utils import projection
from utils import violation_check
from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def prepare_person_data(detections, homography):

    # Create a list to store data related to detections, required for social distancing checks
    # detection_data is a list of tuples where each tuple is of the format
    # (detection_id, transformed_feet_points_array)
    detection_data = []
    for idx, detection in enumerate(detections):
        feet_point = ((detection[0] + detection[2]) / 2, detection[3])
        feet_point = np.array(feet_point, dtype=np.float32)[np.newaxis, :]
        transformed_feet_point = projection.transform_coords(feet_point, homography)

        detection_data.append(
            (idx, {
                'feet_point': transformed_feet_point,
                'bbox': detection}))

    return detection_data


def main():

    # Build the model from a config and model file
    model = init_detector(FLAGS.config_path, FLAGS.model_path, device=FLAGS.device)

    # Calculate the Homography matrix for Perspective Transformation
    homography, distance_threshold = projection.get_homography(
        FLAGS.input_video_path, 1.0)

    # Create Video Capture and Writer objects
    video_capture = cv2.VideoCapture(FLAGS.input_video_path)
    video_writer = cv2.VideoWriter(
        FLAGS.output_video_path,
        cv2.VideoWriter_fourcc(*'VP90'),
        video_capture.get(cv2.CAP_PROP_FPS),
        (960 + 540, 1080))

    # Instantiate a plotting object
    realtime_plot = plots.RealTimePlots()

    # Initialize the Background subtractor for Motion Heatmap generation
    background_subtractor = cv2.createBackgroundSubtractorMOG2()
    video_dim = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2))
    start_time = time.time()
    frame_count = 1

    while True:

        ret, frame = video_capture.read()
        if not ret:
            break

        result = inference_detector(model, frame)[0]

        # Filter the detections with a low confidence threshold
        detections = []
        for box in result:
            if box[4] >= FLAGS.score_thresh:
                detections.append(box[:4].astype(np.int16))

        # Perform social distancing
        person_data = prepare_person_data(detections, homography)
        person_status, person_connections = violation_check.social_distance_check(person_data)

        # Generate Motion Heatmap
        if frame_count == 1:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            heatmap = np.copy(frame)
        else:
            filter = background_subtractor.apply(frame)

            _, th1 = cv2.threshold(filter, 2, 2, cv2.THRESH_BINARY)

            # Create the heatmap visualizations
            accum_image = cv2.add(accum_image, th1)
            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            heatmap = cv2.addWeighted(frame, 0.7, color_image_video, 0.6, 0)

        # Create all visualizations
        bird_view_image = np.zeros(shape=(850, 450, 3))
        frame, bird_view_image = visualize.show_violations(
            frame,
            bird_view_image,
            person_data,
            person_status,
            person_connections)

        realtime_plot.add_data(len(detections), 5)
        image_plot = realtime_plot.retrieve_plot()

        bird_view_image = cv2.resize(bird_view_image, (540, 540))
        frame = cv2.resize(frame, video_dim)
        heatmap = cv2.resize(heatmap, video_dim)

        top_frame = np.concatenate((bird_view_image, frame), axis=1)
        top_frame = np.uint8(top_frame)

        bottom_frame = np.concatenate((image_plot, heatmap), axis=1)
        bottom_frame = np.uint8(bottom_frame)

        output_frame = np.concatenate((top_frame, bottom_frame), axis=0)
        output_frame = np.uint8(output_frame)

        video_writer.write(output_frame)

        logs = 'Frames processed: ' + str(frame_count)
        print('\r' + logs, end='')
        frame_count += 1

    total_time = time.time() - start_time
    time_per_frame = round((total_time / frame_count) * 1000, 2)
    print(f'\nAverage Latency per frame: {time_per_frame} ms')

    video_capture.release()
    video_writer.release()


if __name__ == '__main__':
    FLAGS = get_params.parse_arguments()
    main()

