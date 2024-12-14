
from ultralytics import YOLO
import numpy as np
import json
import os
import argparse
import Detection



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Real-time pose detection using YOLO.")
    parser.add_argument('--model', type=str, required=False, default="yolo11n-pose.pt",
                        help="Path to the YOLO model file.")
    parser.add_argument('--file_path', type=str, required=False, default="records/action_data.json",
                        help="Path to the JSON file for saving pose data.")
    parser.add_argument('--source', type=str, required=False, default="0",
                        help="Source for the input (0 for webcam, or path to video/image file).")
    parser.add_argument('--show', action='store_true',
                        help="Show the detection results in a window.")
    parser.add_argument('--fps', type=int, required=False, default=30,
                        help="Frame rate for the input source.")

    args = parser.parse_args()

    det = Detection.Detector(args.model, args.file_path)
    det.detect(source=args.source, show=args.show, fps=args.fps)


