"""把 SHWD 的 Pascal-VOC 标注转 YOLO 格式, 并按 8:1:1 划分训练/验证/测试.

输入:
  datasets/SHWD/annotations/xmls/*.xml   (Pascal-VOC)
  datasets/SHWD/images/*.png  *.jpg      (原图)

输出:
  datasets/helmet_yolo/
    images/train  val  test/
    labels/train  val  test/
  labels/*.txt 每行: <class_id> <xc> <yc> <w> <h>  (归一化到 0-1)

类映射: helmet=0, head=1  (SHWD 中 person 项实际是没戴帽的头, 不收)
"""
from __future__ import annotations

import random
import shutil
import sys

import cv2

from dataset_utils import compute_splits, find_image, voc_to_yolo
from utils import DATASETS_DIR

CLASS_MAP = {"helmet": 0, "head": 1}


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

    xml_files = sorted(xml_dir.glob("*.xml"))
    print(f"找到 {len(xml_files)} 个 VOC 标注")

    # 图片未解压则提示
    if not any(img_dir.iterdir()):
        for gz in src_root.glob("*.tar.gz"):
            print(f"提示: 你需要先解压 {gz}")
            print("      Windows 上右键 -> 7-Zip -> 解压到当前目录")
        return 1

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

        # SHWD 的 xml 常缺 <size>, 用真实图片尺寸兜底
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                skip += 1
                continue
            ih, iw = img.shape[:2]
        except Exception:
            skip += 1
            continue

        lines = voc_to_yolo(xml, CLASS_MAP, fallback_w=iw, fallback_h=ih)
        if not lines:
            skip += 1
            continue

        shutil.copy2(img_path, out_root / "images" / split / img_path.name)
        (out_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        ok += 1

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(xml_files)}")

    print()
    print(f"[OK] 转换完成: {ok}/{len(xml_files)} 张成功, 跳过 {skip}")
    print(f"     输出: {out_root}")
    print()
    print("下一步: python src/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
