"""GPU / CUDA / PyTorch 环境自检.

打印 torch 版本、CUDA 版本、cuDNN 版本、设备名、显存、
以及是否支持 Blackwell 架构(sm_120), 用于 RTX 5060 的环境验证.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("=" * 60)
    print("  GPU / CUDA / PyTorch 环境自检")
    print("=" * 60)

    try:
        import torch
        import torch.cuda as tc
    except ImportError:
        print("[X] torch 未安装. 请先运行 install.bat")
        return 1

    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA 编译版本: {torch.version.cuda}")
    print(f"cuDNN 版本:    {torch.backends.cudnn.version()}")
    print()

    if not tc.is_available():
        print("[X] CUDA 不可用! 当前是 CPU 版 PyTorch.")
        print("    RTX 5060 必须装 cu128 轮子:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return 2

    n = tc.device_count()
    print(f"检测到 {n} 块 GPU:")
    for i in range(n):
        cap_major, cap_minor = tc.get_device_capability(i)
        props = tc.get_device_properties(i)
        print(f"  [{i}] {tc.get_device_name(i)}")
        print(f"      显存:           {props.total_memory / 1024**3:.1f} GB")
        print(f"      Compute Cap:    sm_{cap_major}{cap_minor}")
        print(f"      多处理器数:     {props.multi_processor_count}")
        print(f"      最大线程/块:    {props.max_threads_per_block}")

    print()
    print("Blackwell sm_120 支持(RTX 50 系): ", end="")
    if cap_major >= 12:  # type: ignore[name-defined]
        print("[OK] 当前 PyTorch 可正确调度 RTX 5060")
    else:
        print("[!] 当前 GPU 不是 Blackwell,但 PyTorch 仍可工作")

    # 实测一次矩阵乘法验证
    print()
    print("实测 GPU 矩阵乘法 (4096x4096, FP16) ...")
    torch.set_default_device("cuda")
    a = torch.randn(4096, 4096, dtype=torch.float16)
    b = torch.randn(4096, 4096, dtype=torch.float16)
    import time
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    tc.synchronize()
    dt = (time.perf_counter() - t0) / 10
    tflops = 2 * 4096**3 / dt / 1e12
    print(f"  单次耗时: {dt*1000:.2f} ms")
    print(f"  实测算力: {tflops:.1f} TFLOPS (FP16)")
    print()
    print("[OK] GPU 完全就绪, 可以开始训练 / 推理.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
