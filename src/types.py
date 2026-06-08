"""运行指南

该模块定义领域数据结构（样本、视角等）。
通常不直接执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiViewSample:
    """一个样本（Sxxxx 文件夹）包含多个视角图像路径。"""

    category: str
    sample_id: str
    view_paths: tuple[Path, ...]

    @property
    def group_folder(self) -> str:
        """返回提交所需的 group_folder 字符串（category/Sxxxx）。"""

        return f"{self.category}/{self.sample_id}"

