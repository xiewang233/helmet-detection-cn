"""训练 YOLOv8 安全帽检测模型.

默认读 configs/train_params.yaml, 也可命令行覆盖.

用法:
  python src/train.py
  python src/train.py --config configs/train_params.yaml --epochs 50 --batch 8
  python src/train.py --model yolov8n.pt --epochs 20   REM 快速试验

输出到 runs/detect/helmet_exp/, 包含:
  weights/best.pt     最优权重
  weights/last.pt     最后一个 epoch
  results.png         训练曲线
  confusion_matrix.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

from utils import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8 训练")
    p.add_argument("--config", default="configs/train_params.yaml")
    p.add_argument("--model", help="覆盖配置里的 model")
    p.add_argument("--data", help="覆盖配置里的 data")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--device", default="0", help="0=GPU, cpu=禁用")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        print(f"[X] 配置文件不存在: {cfg_path}")
        return 1
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # 命令行覆盖
    for k in ("model", "data", "epochs", "batch", "imgsz", "device"):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    # 强制 GPU
    if str(cfg.get("device", "0")).lower() in ("cpu", "-1"):
        print("[!] 本项目强制使用 GPU, device 已重置为 0")
        cfg["device"] = 0

    print("=" * 60)
    print("  YOLOv8 安全帽检测训练")
    print("=" * 60)
    for k, v in cfg.items():
        print(f"  {k:12s}: {v}")
    print()

    # 路径相对于项目根
    data_path = cfg.pop("data")
    if not Path(data_path).is_absolute():
        data_path = str(ROOT / data_path)

    model_path = cfg.pop("model")
    project = cfg.pop("project", None)  # None -> 用 ultralytics 默认 runs/detect
    name = cfg.pop("name", "helmet_exp")
    export_onnx = cfg.pop("export_onnx", False)

    model = YOLO(model_path)
    # 只在显式指定 project 时才传, 否则让 ultralytics 用 settings 的 runs_dir,
    # 避免 save_dir 变成 runs/detect/runs/detect/<name> 双重嵌套
    train_kwargs = dict(data=data_path, name=name, **cfg)
    if project:
        train_kwargs["project"] = project
    results = model.train(**train_kwargs)

    print()
    print(f"[OK] 训练完成")
    print(f"  最佳权重: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"  最后权重: {Path(project) / name / 'weights' / 'last.pt'}")

    if export_onnx:
        print("导出 ONNX ...")
        model.export(format="onnx")

    return 0


if __name__ == "__main__":
    sys.exit(main())
