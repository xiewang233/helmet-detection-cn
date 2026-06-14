"""图片检测.

用法:
  python src/detect_image.py --model yolov8n.pt --source ultralytics/assets/bus.jpg
  python src/detect_image.py --model models/helmet_best.pt --source data/images/test.jpg

参数:
  --model   权重路径, 默认 yolov8n.pt(会自动下载)
  --source  单张图片, 或目录, 或 ultralytics/assets/bus.jpg
  --conf    置信度阈值, 默认 0.25
  --iou     NMS 的 IoU 阈值, 默认 0.7
  --imgsz   输入分辨率, 默认 640
  --half    FP16 推理(更快, RTX 5060 支持)
  --save    保存到 data/output/
  --no-show 不弹窗显示
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

from utils import OUTPUT_DIR, resolve_source


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 图片检测")
    p.add_argument("--model", default="yolov8n.pt", help="权重路径")
    p.add_argument("--source", required=True, help="图片路径或目录")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 推理")
    p.add_argument("--save", action="store_true", help="保存结果到 data/output/")
    p.add_argument("--no-show", action="store_true", help="不弹窗")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.source)

    print(f"模型: {args.model}")
    print(f"输入: {source}")
    print(f"设备: cuda (RTX 5060)")
    print()

    model = YOLO(args.model)

    results = model.predict(
        source=str(source),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        half=args.half,
        device=0,
        show=not args.no_show,
        save=args.save,
        project=str(OUTPUT_DIR),
        name="image_pred",
        exist_ok=True,
    )

    # 汇总
    total = 0
    for r in results:
        n = len(r.boxes)
        total += n
        cls_names = [r.names[int(c)] for c in r.boxes.cls] if n else []
        if n:
            print(f"  {Path(r.path).name}: 检测到 {n} 个 -> {cls_names}")
        else:
            print(f"  {Path(r.path).name}: 无检测")
    print(f"\n共 {len(results)} 张图片, 总检测 {total} 个目标.")

    if args.save:
        print(f"结果已保存到: {OUTPUT_DIR / 'image_pred'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
