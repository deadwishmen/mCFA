"""Utility script for person/body detection using YOLOv8."""

import argparse
from ultralytics import YOLO


def run_yolo_inference(images, weights: str = "yolov8n.pt", save_path: str = "result.jpg"):
    model = YOLO(weights)
    results = model(images)
    for idx, result in enumerate(results):
        out_file = save_path if len(results) == 1 else f"result_{idx}.jpg"
        result.save(filename=out_file)
        print(f"Saved detection output to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 detection on input images")
    parser.add_argument("--images", nargs="+", required=True, help="Paths to input images")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 weights path")
    parser.add_argument("--output", type=str, default="result.jpg", help="Output filename")
    args = parser.parse_args()
    run_yolo_inference(args.images, weights=args.weights, save_path=args.output)
