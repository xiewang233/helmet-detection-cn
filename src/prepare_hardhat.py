"""把 Kaggle Hard Hat Workers 数据集转 YOLO 格式.

输入(Kaggle archive.zip 解压后):
  datasets/HardHat/annotations/hard_hat_workers*.xml   Pascal-VOC
  datasets/HardHat/images/hard_hat_workers*.png        416x416 PNG

输出(8:1:1 划分):
  datasets/helmet_yolo/
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt

类映射:
  helmet  = 0
  head    = 1
  person  = 2
"""
from __future__ import annotations

import random
import sys

from dataset_utils import compute_splits, find_image, link_or_copy, voc_to_yolo
from utils import DATASETS_DIR

CLASS_MAP = {"helmet": 0, "head": 1, "person": 2}


def main() -> int:
    src = DATASETS_DIR / "HardHat"
    xml_dir = src / "annotations"
    img_dir = src / "images"
    if not xml_dir.exists() or not img_dir.exists():
        print(f"[X] 找不到 {xml_dir} 或 {img_dir}")
        return 1

    out = DATASETS_DIR / "helmet_yolo"
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"找到 {len(xml_files)} 个 VOC 标注")

    random.seed(42)
    random.shuffle(xml_files)
    splits = compute_splits(len(xml_files))

    ok = skip = 0
    for i, (xml, split) in enumerate(zip(xml_files, splits)):
        stem = xml.stem
        img_path = find_image(img_dir, stem)
        if img_path is None:
            skip += 1
            continue

        lines = voc_to_yolo(xml, CLASS_MAP)  # HardHat 的 xml 自带 <size>
        if not lines:
            skip += 1
            continue

        # hardlink 省磁盘(同分区才行, 跨分区自动回退 copy)
        link_or_copy(img_path, out / "images" / split / img_path.name)
        (out / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        ok += 1

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(xml_files)}")

    print()
    print(f"[OK] 转换完成: {ok}/{len(xml_files)} 成功, 跳过 {skip}")
    print(f"     输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
