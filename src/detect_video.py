"""视频检测.

用法:
  python src/detect_video.py --model yolov8n.pt --source data/videos/test.mp4
  python src/detect_video.py --model models/helmet_best.pt --source data/videos/test.mp4 --save

参数:
  --model   权重路径
  --source  视频文件
  --conf    置信度阈值
  --iou     NMS IoU
  --imgsz   分辨率
  --half    FP16
  --save    保存标注后的视频到 data/output/video_pred/
  --no-show 不弹窗播放
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import OUTPUT_DIR, resolve_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 视频检测")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--source", required=True, help="视频文件路径")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.source)

    print(f"模型: {args.model}")
    print(f"视频: {source}")
    print()

    model = YOLO(args.model)
    results = model.predict(
        source=str(source),
        stream=True,            # 流式处理, 不一次性把整个视频载入内存
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        device=0,
        show=not args.no_show,
        save=args.save,
        project=str(OUTPUT_DIR),
        name="video_pred",
        exist_ok=True,
        vid_stride=1,           # 每帧都处理
    )

    frame_count = 0
    det_count = 0
    for r in results:
        frame_count += 1
        det_count += len(r.boxes)
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧, 累计检测 {det_count} 个目标")

    print(f"\n共处理 {frame_count} 帧, 总检测 {det_count} 个目标.")
    if args.save:
        print(f"结果视频已保存到: {OUTPUT_DIR / 'video_pred'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
