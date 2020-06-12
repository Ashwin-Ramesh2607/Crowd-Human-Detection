import argparse


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
        default='output.webm',
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
