"""把 Kaggle Hard Hat Workers 数据集转 YOLO 格式.

输入(Kaggle 解压后):
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
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import DATASETS_DIR

CLASS_MAP = {"helmet": 0, "head": 1, "person": 2}


def voc_to_yolo(xml_path: Path) -> tuple[list[str], int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width"))  # type: ignore[arg-type]
    h = int(size.findtext("height"))  # type: ignore[arg-type]

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_MAP:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))  # type: ignore[arg-type]
        ymin = float(bnd.findtext("ymin"))  # type: ignore[arg-type]
        xmax = float(bnd.findtext("xmax"))  # type: ignore[arg-type]
        ymax = float(bnd.findtext("ymax"))  # type: ignore[arg-type]

        xc = (xmin + xmax) / 2.0 / w
        yc = (ymin + ymax) / 2.0 / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h

        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{CLASS_MAP[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines, w, h


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
    n = len(xml_files)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )

    ok = 0
    skip = 0
    for i, (xml, split) in enumerate(zip(xml_files, splits)):
        stem = xml.stem
        img_path = img_dir / f"{stem}.png"
        if not img_path.exists():
            # try jpg
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.exists():
                skip += 1
                continue

        lines, _, _ = voc_to_yolo(xml)
        if not lines:
            skip += 1
            continue

        dst_img = out / "images" / split / img_path.name
        dst_lbl = out / "labels" / split / f"{stem}.txt"
        # 用 hardlink 而不是 copy, 节省磁盘(同分区才能 link, 失败回退 copy)
        try:
            if dst_img.exists():
                dst_img.unlink()
            os_link = getattr(__import__("os"), "link", None)
            if os_link:
                try:
                    os_link(str(img_path), str(dst_img))
                except Exception:
                    shutil.copy2(img_path, dst_img)
            else:
                shutil.copy2(img_path, dst_img)
        except Exception:
            shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok += 1

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{n}")

    print()
    print(f"[OK] 转换完成: {ok}/{n} 成功, 跳过 {skip}")
    print(f"     train/val/test = {n_train}/{n_val}/{n - n_train - n_val}")
    print(f"     输出: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
