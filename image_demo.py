import argparse

import cv2

from mmdet.apis import inference_detector, init_detector, show_result_pyplot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    # test a single image
    img = cv2.imread(args.img)
    result = inference_detector(model, img)[0]
    result = result.astype(np.int16)
    # show the results
    for box in result:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 2)
    cv2.write('test.jpg', img)


if __name__ == '__main__':
    main()
