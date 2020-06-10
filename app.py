import time
import argparse

import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():

    # Build the model from a config and model file
    model = init_detector(FLAGS.config_path, FLAGS.model_path, device=FLAGS.device)

    # Create Video Capture and Writer objects
    video_capture = cv2.VideoCapture(FLAGS.input_video_path)
    video_writer = cv2.VideoWriter(
        FLAGS.output_video_path,
        cv2.VideoWriter_fourcc(*'MP4V'),
        video_capture.get(cv2.CAP_PROP_FPS), 
        (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    start_time = time.time()
    frame_count = 1

    while True:

        ret, frame = video_capture.read()
        if not ret:
            break

        result = inference_detector(model, frame)[0]

        # Visualize the bounding boxes
        for box in result:
            if box[4] >= FLAGS.score_thresh:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 255), 1)

        video_writer.write(frame)

        logs = 'Frames processed: ' + str(frame_count)
        print('\r' + logs, end='')
        frame_count += 1

    total_time = time.time() - start_time
    time_per_frame = round((total_time / frame_count) * 1000, 2)
    print(f'Average Latency per frame: {time_per_frame} ms')

    video_capture.release()
    video_writer.release()


if __name__ == '__main__':
    FLAGS = parse_arguments()
    main()
