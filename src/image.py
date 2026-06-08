"""运行指南

该模块提供图像加载与预处理（统一到 448x448 等）。
通常不直接执行。
"""

from __future__ import annotations

from pathlib import Path

from .errors import DependencyMissingError


def load_rgb_image(path: Path):
    """读取图像为 RGB PIL.Image。"""

    try:
        from PIL import Image
    except Exception as exc:
        raise DependencyMissingError("缺少依赖 Pillow，请先安装: pip install pillow") from exc

    img = Image.open(path)
    return img.convert("RGB")


def resize_image(img, *, size: int):
    """将 PIL.Image 缩放到固定正方形尺寸。"""

    return img.resize((size, size))

