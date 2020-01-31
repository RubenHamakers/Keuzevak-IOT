import tensorflow as tf
import cv2
import time
import argparse

import posenet
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(50, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture('./640.mp4')
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        avg = 0
        people = 0
        while cap.isOpened():

            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )


            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=20,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            people += len([i for i in pose_scores if i > 0.15])
            frame_count += 1
            print("Average people so far: " + str(people/frame_count))

            if(frame_count%150==0):
                print('Average FPS: ', frame_count / (time.time() - start))
                print("Sending GET request with avg people.")
                r = requests.get(url='https://api.thingspeak.com/update?api_key=M8ND4LO11PWFJU0Y&field1=' +str(people/frame_count))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        


if __name__ == "__main__":
    main()