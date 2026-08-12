"""公共工具: 路径、颜色、绘制."""
from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认 GBK, 中文输出会乱码; 统一切 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 项目根
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = DATA_DIR / "output"
DATASETS_DIR = ROOT / "datasets"

# 确保 output 目录存在
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 安全帽场景配色 (BGR for OpenCV)
COLORS = {
    "helmet":  (0, 255, 0),     # 绿色: 戴了安全帽
    "head":    (0, 0, 255),     # 红色: 没戴
    "person":  (255, 200, 0),   # 橙色: 人
    "default": (0, 255, 255),   # 黄色
}


def resolve_source(path: str | Path) -> Path:
    """把命令行传入的路径解析成绝对路径.

    支持:
      - 绝对路径
      - 项目相对路径
      - ultralytics 内置资源(如 ultralytics/assets/bus.jpg)
    """
    p = Path(path)
    if p.exists():
        return p.resolve()

    # 项目根相对
    p2 = ROOT / path
    if p2.exists():
        return p2.resolve()

    # ultralytics 自带资源 (兼容 "ultralytics/assets/x.jpg" 和 "assets/x.jpg")
    try:
        import ultralytics
        ultra_root = Path(ultralytics.__file__).resolve().parent
        rel = str(path)
        if rel.startswith("ultralytics/"):
            rel = rel[len("ultralytics/"):]
        p3 = ultra_root / rel
        if p3.exists():
            return p3.resolve()
    except ImportError:
        pass

    print(f"[!] 找不到文件: {path}", file=sys.stderr)
    sys.exit(2)


def color_for(name: str):
    return COLORS.get(name, COLORS["default"])
