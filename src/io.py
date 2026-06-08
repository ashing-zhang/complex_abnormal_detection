"""运行指南

该模块负责写出 submission.csv、mask png 以及最终 zip 包。
通常不直接执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .errors import DependencyMissingError


def ensure_dir(path: Path) -> None:
    """确保目录存在。"""

    path.mkdir(parents=True, exist_ok=True)


def write_submission_csv(path: Path, rows: Iterable[tuple[str, float]]) -> None:
    """写出 submission.csv。"""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("group_folder,anomaly_score\n")
        for group_folder, score in rows:
            f.write(f"{group_folder},{score:.6f}\n")


def save_grayscale_png(path: Path, *, array_2d_uint8) -> None:
    """保存单通道灰度 png。"""

    try:
        from PIL import Image
    except Exception as exc:
        raise DependencyMissingError("缺少依赖 Pillow，请先安装: pip install pillow") from exc

    ensure_dir(path.parent)
    img = Image.fromarray(array_2d_uint8, mode="L")
    img.save(path)


def make_submission_zip(zip_path: Path, *, submission_csv: Path, predicted_masks_dir: Path) -> None:
    """打包 submission.csv 与 predicted_masks/ 为 zip。"""

    import zipfile

    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_csv, arcname=submission_csv.name)
        for file_path in _iter_files(predicted_masks_dir):
            arc = str(file_path.relative_to(predicted_masks_dir.parent)).replace("\\", "/")
            zf.write(file_path, arcname=arc)


def _iter_files(root: Path) -> Iterable[Path]:
    """递归遍历文件。"""

    for p in root.rglob("*"):
        if p.is_file():
            yield p

