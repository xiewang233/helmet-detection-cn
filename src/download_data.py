"""下载 SHWD (Safety Helmet Wearing Dataset).

数据集来源:
  https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset
  作者: njvisionpower, MIT License
  规模: 7581 张图, 9044 个标注(其中 helmet 4957, person 4087)

下载到 datasets/SHWD/, 包含:
  - annotations/xmls/      Pascal-VOC XML 标注
  - images/                原图
后续用 prepare_data.py 转 YOLO 格式.

如果 GitHub 直连慢, 可设置环境变量 GITHUB_MIRROR 用代理, 如:
  set GITHUB_MIRROR=https://ghproxy.com/https://github.com
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from utils import DATASETS_DIR

# SHWD 数据集两个压缩包
URLS = {
    "annotations": (
        "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/"
        "raw/master/dataset/annotations.tar.gz"
    ),
    "images": (
        "https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/"
        "raw/master/dataset/images.tar.gz"
    ),
}

MIRROR = os.environ.get("GITHUB_MIRROR", "").rstrip("/")


def progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "=" * int(pct // 2)
        sys.stdout.write(f"\r  [{bar:<50}] {pct:5.1f}%")
        sys.stdout.flush()


def download(url: str, dst: Path) -> None:
    full_url = f"{MIRROR}/{url}" if MIRROR else url
    print(f"  从 {full_url}")
    print(f"  到 {dst}")
    urllib.request.urlretrieve(full_url, dst, reporthook=progress)
    print()
    print(f"  大小: {dst.stat().st_size / 1024 / 1024:.1f} MB")


def main() -> int:
    target = DATASETS_DIR / "SHWD"
    target.mkdir(parents=True, exist_ok=True)

    print(f"下载 SHWD 安全帽数据集到: {target}")
    print()

    for name, url in URLS.items():
        dst = target / f"{name}.tar.gz"
        if dst.exists() and dst.stat().st_size > 1024:
            print(f"[skip] {dst} 已存在")
            continue
        try:
            download(url, dst)
        except Exception as e:
            print(f"\n[X] 下载 {name} 失败: {e}")
            print("    可以手动从浏览器下载, 放到:")
            print(f"    {dst}")
            print("    或者设置 GITHUB_MIRROR 环境变量后重试:")
            print('    set GITHUB_MIRROR=https://ghproxy.com/https://github.com')
            return 1

    print()
    print("[OK] 下载完成. 接下来运行:")
    print("  python src/prepare_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
