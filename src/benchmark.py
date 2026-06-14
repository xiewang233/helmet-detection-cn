"""在 RTX 5060 上跑 YOLOv8 各模型大小的性能压测.

测试指标:
  - 推理延迟 (ms)
  - 吞吐量 (FPS)
  - 显存占用 (MB)

用法:
  python src/benchmark.py
  python src/benchmark.py --models yolov8n.pt yolov8s.pt --half
  python src/benchmark.py --imgsz 416
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTX 5060 YOLOv8 性能压测")
    p.add_argument(
        "--models", nargs="+",
        default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="要测的模型列表",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16 推理")
    p.add_argument("--warmup", type=int, default=10, help="预热轮数")
    p.add_argument("--iters", type=int, default=100, help="正式测试轮数")
    return p.parse_args()


def benchmark_one(
    name: str, imgsz: int, half: bool, warmup: int, iters: int
) -> dict:
    """对一个模型做 forward 压测, 返回统计."""
    print(f"\n>>> 加载 {name} ...")
    model = YOLO(name)
    device = "cuda:0"

    # 预热(GPU 需要预热才能稳定, 否则第一次 forward 会触发 kernel JIT)
    # 用 BCHW float tensor
    dummy = torch.rand(1, 3, imgsz, imgsz, device=device, dtype=torch.float32)
    for _ in range(warmup):
        model.predict(
            source=dummy, imgsz=imgsz, half=half, device=device, verbose=False
        )
    torch.cuda.synchronize()

    # 显存峰值(GPU)
    torch.cuda.reset_peak_memory_stats(device)

    # 正式测
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(
            source=dummy, imgsz=imgsz, half=half, device=device, verbose=False
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times.sort()
    # 去掉最快最慢各 5%(更稳)
    trim_n = max(1, int(len(times) * 0.05))
    trimmed = times[trim_n:-trim_n]
    mean_ms = sum(trimmed) / len(trimmed) * 1000
    fps = len(trimmed) / sum(trimmed)
    vram_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024

    # 文件大小
    from pathlib import Path
    size_mb = Path(name).stat().st_size / 1024 / 1024 if Path(name).exists() else 0

    return {
        "name": name,
        "size_mb": size_mb,
        "mean_ms": mean_ms,
        "fps": fps,
        "vram_mb": vram_mb,
    }


def main() -> int:
    args = parse_args()

    # 表头
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
            r = benchmark_one(
                name, args.imgsz, args.half, args.warmup, args.iters
            )
            results.append(r)
        except Exception as e:
            print(f"  [!] {name} 跑失败: {e}")

    # 汇总表
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
