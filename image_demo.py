import argparse

import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def parse_arguments():

    parser = argparse.ArgumentParser(description='Social Distancing AI')

    parser.add_argument(
        '--input_video_path',
        type=str,
        default='',
        help='Path to the input video file')

    parser.add_argument(
        '--output_video_path',
        type=str,
        default='output.mp4',
        help='Path to the output video file')

    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py',
        help='Path to the config file for the required model')

    parser.add_argument(
        '--model_path',
        type=str,
        default='crowd_human_full_faster_rcnn_r50_fpn_2x.pth',
        help='Path to the trained model for inference')

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device name and number to be used for inference')

    parser.add_argument(
        '--score_thresh',
        type=float,
        default=0.3,
        help='Score threshold above which to display bounding boxes')

    return parser.parse_args()


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

    video_capture.release()
    video_writer.release()


if __name__ == '__main__':
    FLAGS = parse_arguments()
    main()
