"""运行指南

该模块实现简单的最近邻相似度搜索（基于余弦相似度）。
通常不直接执行。
"""

from __future__ import annotations

from typing import Any


def max_cosine_similarity(
    query: Any,
    bank: Any,
    *,
    chunk_size: int,
) -> Any:
    """计算每个 query 向量在 bank 中的最大余弦相似度。"""

    import torch

    if query.ndim != 2 or bank.ndim != 2:
        raise ValueError("query/bank 必须为二维张量")
    if query.shape[1] != bank.shape[1]:
        raise ValueError("query/bank 的特征维度必须一致")

    max_sim = None
    for start in range(0, bank.shape[0], chunk_size):
        part = bank[start : start + chunk_size]
        sim = query @ part.T
        part_max = sim.max(dim=1).values
        max_sim = part_max if max_sim is None else torch.maximum(max_sim, part_max)

    return max_sim

