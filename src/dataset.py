"""运行指南

该模块负责扫描数据集目录并生成多视角样本列表。
通常不直接执行。
"""

from __future__ import annotations

from pathlib import Path

from .errors import DatasetStructureError
from .types import MultiViewSample


def scan_multiview_samples(root_dir: Path, *, view_count: int) -> list[MultiViewSample]:
    """扫描 root_dir 下的 {category}/{Sxxxx}/{0..4}.png 样本结构。"""

    if not root_dir.exists():
        raise DatasetStructureError(f"目录不存在: {root_dir}")

    samples: list[MultiViewSample] = []
    for category_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        for sample_dir in sorted([p for p in category_dir.iterdir() if p.is_dir()]):
            view_paths = _collect_view_paths(sample_dir, view_count=view_count)
            samples.append(
                MultiViewSample(
                    category=category_dir.name,
                    sample_id=sample_dir.name,
                    view_paths=view_paths,
                )
            )
    return samples


def group_by_category(samples: list[MultiViewSample]) -> dict[str, list[MultiViewSample]]:
    """将样本按类别分组。"""

    grouped: dict[str, list[MultiViewSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    return grouped


def _collect_view_paths(sample_dir: Path, *, view_count: int) -> tuple[Path, ...]:
    """收集并校验单个样本的多视角图像路径。"""

    view_paths: list[Path] = []
    for view_index in range(view_count):
        p = sample_dir / f"{view_index}.png"
        if not p.exists():
            raise DatasetStructureError(f"缺少视角图像: {p}")
        view_paths.append(p)
    return tuple(view_paths)

