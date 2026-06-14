"""把终端文本输出渲染成 PNG 截图风格.

用法:
  python src/render_terminal.py --input /tmp/gpu_out.txt --output screenshots/00_check_gpu.png --title "GPU 自检"

读取终端输出文件, 模拟黑底绿字终端样式, 输出 PNG.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ANSI 颜色码 -> RGB
ANSI_COLORS = {
    "30": (80, 80, 80),     # black
    "31": (255, 90, 90),    # red
    "32": (110, 230, 110),  # green
    "33": (230, 200, 90),   # yellow
    "34": (110, 160, 255),  # blue
    "35": (220, 130, 220),  # magenta
    "36": (110, 220, 220),  # cyan
    "37": (220, 220, 220),  # white
}
DEFAULT_FG = (220, 220, 220)


def strip_ansi(text: str) -> tuple[str, list]:
    """去掉 ANSI 控制码, 返回纯文本 + 每行每段的着色信息."""
    # 把光标控制字符(\x1b[K, \r 等)先去掉
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)
    text = text.replace("\r", "\n")
    return text


def find_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """找系统等宽字体, Windows 优先 Consolas."""
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/lucon.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def render(text: str, title: str, output: Path) -> None:
    # 清洗
    text = strip_ansi(text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    # 去掉连续空行
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                cleaned.append(ln)
        else:
            blank_run = 0
            cleaned.append(ln)
    lines = cleaned

    font = find_font(16)
    title_font = find_font(18)

    # 测量
    dummy = Image.new("RGB", (10, 10))
    dd = ImageDraw.Draw(dummy)
    line_h = 22
    char_w = 9
    max_w = max((dd.textlength(ln, font=font) for ln in lines), default=0)
    width = int(max_w) + 60
    height = line_h * (len(lines) + 3) + 30

    # 创建画布
    img = Image.new("RGB", (width, height), (30, 30, 35))
    draw = ImageDraw.Draw(img)

    # 顶部标题栏(模拟 macOS 风格)
    draw.rectangle([0, 0, width, 36], fill=(50, 50, 55))
    # 三个圆点
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        cx = 22 + i * 26
        draw.ellipse([cx - 9, 12, cx + 9, 30], fill=color)
    # 标题居中
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 9), title, fill=(200, 200, 200), font=title_font)

    # 终端内容
    y = 50
    for ln in lines:
        # 简单语法着色: 关键字绿, 数字蓝, 错误红, [OK]/[!] 高亮
        color = DEFAULT_FG
        s = ln
        if "[OK]" in s:
            color = (110, 230, 110)
        elif "[X]" in s or "[!]" in s or "Error" in s or "fail" in s.lower():
            color = (255, 130, 130)
        elif "PyTorch" in s or "CUDA" in s or "cuDNN" in s or "Python" in s:
            color = (110, 200, 255)
        elif "NVIDIA" in s or "RTX" in s:
            color = (200, 180, 90)
        elif re.search(r"\b\d+(\.\d+)?\b", s) and ("=" in s or ":" in s):
            color = (220, 200, 130)
        draw.text((20, y), s, fill=color, font=font)
        y += line_h

    img.save(output, "PNG")
    print(f"  saved: {output}  ({width}x{height})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="Terminal")
    args = p.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    render(text, args.title, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
