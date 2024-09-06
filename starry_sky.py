#!/usr/bin/env python3
import argparse
import cv2
from lib.oakd_yolo_star import OakdYoloStar
from lib.akari_yolo_lib.util import download_file


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--fps",
        help="Camera frame fps. This should be smaller than nn inference fps",
        default=7,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--display_camera",
        help="Display camera rgb and depth frame",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--robot_coordinate",
        help="Convert object pos from camera coordinate to robot coordinate",
        action="store_true",
    )
    parser.add_argument(
        "--log_path",
        help="Path to save orbit data",
        type=str,
        default="log/"
    )
    args = parser.parse_args()
    bird_frame = True
    model_path = "model/human_parts.blob"
    config_path = "config/human_parts.json"
    download_file(
        model_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.blob",
    )
    download_file(
        config_path,
        "https://github.com/AkariGroup/akari_yolo_models/raw/main/human_parts/human_parts.json",
    )
    end = False
    while not end:
        oakd_yolo_star = OakdYoloStar(
            config_path="config/human_parts.json",
            model_path="model/human_parts.blob",
            fps=args.fps,
            cam_debug=args.display_camera,
            robot_coordinate=args.robot_coordinate,
            track_targets=["person"],
            show_bird_frame=bird_frame,
            show_spatial_frame=False,
            show_orbit=True,
            log_path=args.log_path,
        )
        oakd_yolo_star.update_bird_frame_width(8000)
        oakd_yolo_star.update_bird_frame_distance(10000)
        is_fullscreen = True
        cv2.namedWindow("birds",cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("birds",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        while True:
            frame = None
            detections = []
            try:
                frame, detections, tracklets = oakd_yolo_star.get_frame()
            except BaseException:
                print("===================")
                print("get_frame() error! Reboot OAK-D.")
                print("If reboot occur frequently, Bandwidth may be too much.")
                print("Please lower FPS.")
                print("==================")
                break
            if frame is not None:
                oakd_yolo_star.display_frame("nn", frame, tracklets)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                end = True
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty('birds', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('birds', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        oakd_yolo_star.close()


if __name__ == "__main__":
    main()
