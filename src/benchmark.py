"""在 RTX 5060 上跑 YOLOv8 各模型大小的性能压测.

测试指标:
  - 推理延迟 (ms, 已 trim 5% 极值)
  - 吞吐量 (FPS)
  - 显存峰值 (MB)
  - 权重大小 (MB)

结果同时打印终端表格, 并可选落盘:
  --out <dir>  生成 benchmark.csv + benchmark.png (延迟/FPS/显存 三柱图)

用法:
  python src/benchmark.py
  python src/benchmark.py --models yolov8n.pt yolov8s.pt --half
  python src/benchmark.py --imgsz 416 --out runs/benchmark
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from utils import ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTX 5060 YOLOv8 性能压测")
    p.add_argument(
        "--models", nargs="+",
        default=["yolov8n.pt", "yolov8s.pt"],
        help="要测的模型列表",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 推理")
    p.add_argument("--warmup", type=int, default=10, help="预热轮数")
    p.add_argument("--iters", type=int, default=100, help="正式测试轮数")
    p.add_argument("--out", default="runs/benchmark", help="结果落盘目录(设为空串则只打印)")
    return p.parse_args()


def benchmark_one(name: str, imgsz: int, half: bool, warmup: int, iters: int) -> dict:
    """对一个模型做 forward 压测, 返回统计 dict."""
    print(f"\n>>> 加载 {name} ...")
    model = YOLO(name)
    device = "cuda:0"

    # 预热(GPU 首次 forward 会 JIT kernel, 必须预热才稳)
    dummy = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    for _ in range(warmup):
        model.predict(source=dummy, imgsz=imgsz, half=half, device=device, verbose=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    # 正式测
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(source=dummy, imgsz=imgsz, half=half, device=device, verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    # trim 5% 极值更稳
    times.sort()
    trim_n = max(1, int(len(times) * 0.05))
    trimmed = times[trim_n:-trim_n]
    mean_ms = sum(trimmed) / len(trimmed) * 1000
    fps = len(trimmed) / sum(trimmed)
    vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    size_mb = Path(name).stat().st_size / 1024 / 1024 if Path(name).exists() else 0.0

    # 测完释放, 避免多模型测试时显存累积
    del model
    torch.cuda.empty_cache()

    return {
        "name": name, "size_mb": size_mb, "mean_ms": mean_ms,
        "fps": fps, "vram_mb": vram_mb,
    }


def save_csv(results: list[dict], csv_path: Path) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "size_mb", "latency_ms", "fps", "vram_mb"])
        for r in results:
            w.writerow([r["name"], f"{r['size_mb']:.1f}", f"{r['mean_ms']:.2f}",
                        f"{r['fps']:.1f}", f"{r['vram_mb']:.0f}"])


def save_plot(results: list[dict], imgsz: int, half: bool, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["name"] for r in results]
    metrics = [
        ("Latency (ms)", [r["mean_ms"] for r in results], "#4C72B0"),
        ("FPS",          [r["fps"]      for r in results], "#55A868"),
        ("VRAM (MB)",    [r["vram_mb"]  for r in results], "#C44E52"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (title, vals, color) in zip(axes, metrics):
        ax.bar(names, vals, color=color)
        ax.set_title(title)
        ax.set_xticklabels(names, rotation=15, ha="right")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    half_tag = "FP16" if half else "FP32"
    fig.suptitle(
        f"YOLOv8 benchmark @ {imgsz}px ({half_tag}) on {torch.cuda.get_device_name(0)}"
    )
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    print("=" * 78)
    print("  RTX 5060 YOLOv8 性能压测")
    print("=" * 78)
    print(
        f"  设备: {torch.cuda.get_device_name(0)}, "
        f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
    )
    print(
        f"  配置: imgsz={args.imgsz}, half={args.half}, "
        f"warmup={args.warmup}, iters={args.iters}"
    )
    print()

    results = []
    for name in args.models:
        try:
            results.append(benchmark_one(name, args.imgsz, args.half, args.warmup, args.iters))
        except Exception as e:
            print(f"  [!] {name} 跑失败: {e}")

    # 终端汇总表
    print()
    print("=" * 78)
    print("  压测结果汇总")
    print("=" * 78)
    hdr = f"  {'Model':<14} {'Size(MB)':<10} {'Latency(ms)':<14} {'FPS':<10} {'VRAM(MB)':<10}"
    print(hdr)
    print("  " + "-" * 60)
    for r in results:
        print(
            f"  {r['name']:<14} {r['size_mb']:<10.1f} "
            f"{r['mean_ms']:<14.2f} {r['fps']:<10.1f} {r['vram_mb']:<10.0f}"
        )
    print()
    print("  注: 延迟已 trim 5% 极值. 显存是峰值. 实际受其他进程影响.")

    # 落盘 CSV + PNG
    if args.out and results:
        out_dir = ROOT / args.out
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "benchmark.csv"
        png_path = out_dir / "benchmark.png"
        save_csv(results, csv_path)
        save_plot(results, args.imgsz, args.half, png_path)
        print(f"\n  CSV: {csv_path}")
        print(f"  PNG: {png_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
