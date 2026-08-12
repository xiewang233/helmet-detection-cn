"""数据集准备公共工具.

prepare_hardhat.py 和 prepare_data.py 共享的逻辑(原本两份高度重复, 现抽到一处):
  - voc_to_yolo:    Pascal-VOC XML → YOLO 归一化坐标
  - compute_splits: train/val/test 按 8:1:1 划分(配合 random.seed 可复现)
  - link_or_copy:   hardlink 优先, 失败回退 copy(省磁盘)
  - find_image:     按 stem 在目录里找 png/jpg/jpeg
"""
from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = (".png", ".jpg", ".jpeg")


def voc_to_yolo(
    xml_path: Path,
    class_map: dict[str, int],
    fallback_w: int | None = None,
    fallback_h: int | None = None,
) -> list[str]:
    """读 VOC xml, 返回 YOLO 行列表 ['cls xc yc w h', ...], 坐标归一化到 0-1.

    图尺寸优先取 xml 内 <size>; 缺失或为 0 时用 fallback_w/fallback_h
    (例如从真实图片读出来的). 坐标全部 clamp 到 [0, 1] 防越界.
    """
    root = ET.parse(xml_path).getroot()

    w: int | None = None
    h: int | None = None
    size = root.find("size")
    if size is not None:
        w = _to_int(size.findtext("width"))
        h = _to_int(size.findtext("height"))
    w = w or fallback_w
    h = h or fallback_h
    if not w or not h:
        return []  # 没有尺寸无法归一化

    lines: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in class_map:
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

        lines.append(
            f"{class_map[name]} "
            f"{max(0.0, min(1.0, xc)):.6f} {max(0.0, min(1.0, yc)):.6f} "
            f"{max(0.0, min(1.0, bw)):.6f} {max(0.0, min(1.0, bh)):.6f}"
        )
    return lines


def compute_splits(
    n: int, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> list[str]:
    """生成长度 n 的 split 标签列表(train/val/test), 默认 8:1:1.

    调用方应先 random.seed + random.shuffle(文件列表) 再 zip(splits),
    这样划分既固定又可复现.
    """
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        ["train"] * n_train
        + ["val"] * n_val
        + ["test"] * (n - n_train - n_val)
    )


def link_or_copy(src: Path, dst: Path) -> None:
    """dst 已存在先删; 优先 hardlink(同分区省磁盘), 失败回退 copy2."""
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except (OSError, AttributeError):
        shutil.copy2(src, dst)


def find_image(img_dir: Path, stem: str) -> Path | None:
    """按 stem 依次找 .png/.jpg/.jpeg, 命中即返回, 都没有返回 None."""
    for ext in IMG_EXTS:
        cand = img_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def _to_int(text: str | None) -> int | None:
    try:
        return int(text) if text else None
    except (TypeError, ValueError):
        return None
