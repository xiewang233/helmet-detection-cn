"""把 SHWD 的 Pascal-VOC 标注转 YOLO 格式, 并按 8:1:1 划分训练/验证/测试.

输入:
  datasets/SHWD/annotations/xmls/*.xml   (Pascal-VOC)
  datasets/SHWD/images/*.png  *.jpg      (原图)

输出:
  datasets/helmet_yolo/
    images/train  val  test/
    labels/train  val  test/
  labels/*.txt 每行: <class_id> <xc> <yc> <w> <h>  (归一化到 0-1)

类映射: helmet=0, head=1  (SHWD 中 person 项实际上是没戴帽的头)
"""
from __future__ import annotations

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import DATASETS_DIR

# SHWD 原标注里的类名
CLASS_MAP = {
    "helmet": 0,
    "head": 1,
}


def voc_to_yolo(xml_path: Path, img_w: int, img_h: int) -> list[str] | None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines: list[str] = []

    size = root.find("size")
    if size is not None:
        w = int(size.findtext("width", default=str(img_w)))  # type: ignore[arg-type]
        h = int(size.findtext("height", default=str(img_h)))  # type: ignore[arg-type]
        img_w, img_h = w, h

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

        # 归一化到 0-1
        xc = (xmin + xmax) / 2.0 / img_w
        yc = (ymin + ymax) / 2.0 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h

        # 防越界
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        lines.append(f"{CLASS_MAP[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def main() -> int:
    src_root = DATASETS_DIR / "SHWD"
    xml_dir = src_root / "annotations" / "xmls"
    # SHWD 解压后图片在 images/ 或直接在 SHWD/ 下, 兼容两种情况
    img_dir = (src_root / "images") if (src_root / "images").exists() else src_root

    if not xml_dir.exists():
        print(f"[X] 找不到标注目录: {xml_dir}")
        print("    请先运行: python src/download_data.py")
        return 1

    out_root = DATASETS_DIR / "helmet_yolo"
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 收集所有 xml
    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"找到 {len(xml_files)} 个 VOC 标注")

    # 解压图片: 如果有 .tar.gz 还没解, 提示用户
    if not any(img_dir.iterdir()):
        for gz in src_root.glob("*.tar.gz"):
            print(f"提示: 你需要先解压 {gz}")
            print(f"  在 Windows 上右键 -> 7-Zip -> 解压到当前目录")
        return 1

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
    for i, (xml, split) in enumerate(zip(xml_files, splits)):
        stem = xml.stem
        # 找对应图片(.png or .jpg)
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            cand = img_dir / f"{stem}{ext}"
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        # 读图片尺寸
        try:
            import cv2
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            ih, iw = img.shape[:2]
        except Exception:
            continue

        # 转 YOLO txt
        lines = voc_to_yolo(xml, iw, ih)
        if not lines:
            continue

        # 复制图片
        dst_img = out_root / "images" / split / img_path.name
        dst_lbl = out_root / "labels" / split / f"{stem}.txt"
        shutil.copy2(img_path, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ok += 1

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{n}")

    print()
    print(f"[OK] 转换完成: {ok}/{n} 张成功")
    print(f"     输出: {out_root}")
    print(f"     train/val/test = {n_train}/{n_val}/{n - n_train - n_val}")
    print()
    print("下一步: python src/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
