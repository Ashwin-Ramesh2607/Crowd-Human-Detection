import argparse

import cv2
import numpy as np

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def parse_arguments():

    parser = argparse.ArgumentParser(description='Social Distancing AI')

    parser.add_argument(
        '--image_path',
        type=str,
        default='',
        help='Path to the demo image file')

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

    # build the model from a config file and a checkpoint file
    model = init_detector(FLAGS.config_path, FLAGS.model_path, device=FLAGS.device)

    # test a single image
    image = cv2.imread(FLAGS.image_path)
    result = inference_detector(model, image)[0]

    # show the results
    for box in result:
        if box[4] >= FLAGS.score_thresh:
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

    cv2.imwrite('test.jpg', image)


if __name__ == '__main__':
    FLAGS = parse_arguments()
    main()
