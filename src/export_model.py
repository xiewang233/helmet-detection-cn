"""模型导出.

把 .pt 训练权重导出为其他格式, 用于部署加速.
RTX 5060 (Blackwell) 上推荐导 ONNX 或 TensorRT.

用法:
  python src/export_model.py --weights models/helmet_best.pt --format onnx
  python src/export_model.py --weights models/helmet_best.pt --format engine --half

支持的格式:
  onnx        ONNX, 通用, 推理引擎可以用 onnxruntime
  engine      TensorRT, NVIDIA GPU 专用, 最快
  openvino    Intel CPU 集成
  coreml      Apple
  tflite      移动端
"""
from __future__ import annotations

import argparse
import sys

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 模型导出")
    p.add_argument("--weights", required=True, help=".pt 权重路径")
    p.add_argument(
        "--format", default="onnx",
        choices=["onnx", "engine", "openvino", "coreml", "tflite", "torchscript"],
    )
    p.add_argument("--half", action="store_true", help="FP16 量化(RTX 5060 推荐)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dynamic", action="store_true", help="动态 batch")
    p.add_argument("--simplify", action="store_true", help="简化 ONNX 图")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(f"导出: {args.weights} -> {args.format}")
    print()

    model = YOLO(args.weights)
    try:
        out = model.export(
            format=args.format,
            half=args.half,
            imgsz=args.imgsz,
            batch=args.batch,
            dynamic=args.dynamic,
            simplify=args.simplify,
            device=0,
        )
    except Exception as e:
        print(f"[X] 导出失败: {e}")
        if args.format == "engine":
            print("    TensorRT 导出需要 tensorrt 包, 且要和 CUDA 版本匹配:")
            print("    pip install tensorrt   (或从 NVIDIA 官网下对应版本)")
        elif args.format == "onnx":
            print("    ONNX 导出需要 onnx 包: pip install onnx")
            print("    --simplify 还需要 onnxsim: pip install onnxsim")
        return 1

    print()
    print(f"[OK] 导出完成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
