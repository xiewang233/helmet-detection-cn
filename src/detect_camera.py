"""摄像头实时检测.

按 q 退出.

用法:
  python src/detect_camera.py --model yolov8n.pt
  python src/detect_camera.py --model models/helmet_best.pt --cam 0 --half

参数:
  --model   权重路径
  --cam     摄像头索引, 默认 0
  --conf    置信度阈值
  --iou     NMS IoU
  --imgsz   分辨率
  --half    FP16
  --width   请求的摄像头分辨率宽
  --height  请求的摄像头分辨率高
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 摄像头实时检测")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)  # Windows 用 DSHOW 后端
    if not cap.isOpened():
        print(f"[X] 无法打开摄像头 {args.cam}")
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"模型: {args.model}")
    print(f"摄像头: {args.cam}, 请求 {args.width}x{args.height}")
    print(f"实际: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("按 q 退出, 按 s 截图")
    print()

    model = YOLO(args.model)
    fps_t0 = time.perf_counter()
    frames = 0
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[X] 读帧失败")
            break

        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            half=args.half,
            device=0,
            verbose=False,
        )
        annotated = results[0].plot()

        # FPS 计算
        frames += 1
        if frames >= 10:
            now = time.perf_counter()
            fps = frames / (now - fps_t0)
            frames = 0
            fps_t0 = now

        cv2.putText(
            annotated, f"FPS: {fps:.1f}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2,
        )
        cv2.imshow("YOLOv8 Camera (press q to quit)", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            from utils import OUTPUT_DIR
            out = OUTPUT_DIR / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(out), annotated)
            print(f"  截图已保存: {out}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
