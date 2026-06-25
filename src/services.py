"""运行指南

该模块实现核心用例：
1) 从 Train 正常样本构建类别内 patch 特征库（Memory Bank）
2) 对 Test_A 输出 anomaly_score 与 5 视角 mask

通常由入口模块调用，不直接执行。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import AppConfig
from .dataset import group_by_category, scan_multiview_samples
from .feature_extractor import FeatureExtractor, PatchFeatures, build_feature_extractor
from .image import load_rgb_image, resize_image
from .io import ensure_dir, make_submission_zip, save_grayscale_png, write_submission_csv
from .nn import max_cosine_similarity
from .types import MultiViewSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Paths:
    """输出路径集合。"""

    submission_csv: Path
    predicted_masks_dir: Path
    zip_path: Path
    cache_dir: Path


def run_prediction_pipeline(cfg: AppConfig) -> Paths:
    """运行完整流程并返回产物路径。"""

    paths = _build_paths(cfg)
    ensure_dir(paths.predicted_masks_dir)
    ensure_dir(paths.cache_dir)

    extractor = build_feature_extractor(cfg.feature_extractor)

    train_samples = scan_multiview_samples(cfg.dataset.train_dir, view_count=cfg.dataset.view_count)
    test_samples = scan_multiview_samples(cfg.dataset.test_dir, view_count=cfg.dataset.view_count)
    test_categories = {s.category for s in test_samples}
    logger.info("Train 样本数: %s", len(train_samples))
    logger.info("Test 样本数: %s", len(test_samples))
    logger.info("Test 类别数: %s", len(test_categories))

    banks = _build_memory_banks(
        cfg=cfg,
        extractor=extractor,
        train_samples=train_samples,
        test_categories=test_categories,
        cache_dir=paths.cache_dir,
    )

    rows = _predict_and_write_masks(
        cfg=cfg,
        extractor=extractor,
        banks=banks,
        test_samples=test_samples,
        predicted_masks_dir=paths.predicted_masks_dir,
    )

    write_submission_csv(paths.submission_csv, rows)
    make_submission_zip(paths.zip_path, submission_csv=paths.submission_csv, predicted_masks_dir=paths.predicted_masks_dir)
    return paths


def _build_paths(cfg: AppConfig) -> Paths:
    """根据配置构建输出路径。"""

    work = cfg.output.work_dir
    return Paths(
        submission_csv=work / cfg.output.submission_csv_name,
        predicted_masks_dir=work / cfg.output.predicted_masks_dir_name,
        zip_path=work / cfg.output.zip_name,
        cache_dir=work / "cache",
    )


def _build_memory_banks(
    *,
    cfg: AppConfig,
    extractor: FeatureExtractor,
    train_samples: list[MultiViewSample],
    test_categories: set[str],
    cache_dir: Path,
) -> dict[str, Any]:
    """为测试集涉及的类别构建 patch 特征库，并生成全局兜底库。"""

    if not train_samples:
        raise RuntimeError("未发现任何训练样本，无法构建特征库")

    grouped = group_by_category(train_samples)
    banks: dict[str, Any] = {}
    for category in sorted(test_categories):
        samples = grouped.get(category)
        if not samples:
            logger.warning("Train 中缺少类别 %s，推理将使用全局特征库", category)
            continue
        bank = _load_bank_from_cache(cache_dir, category) if cfg.memory_bank.cache_enabled else None
        if bank is None:
            logger.info("构建类别特征库: %s", category)
            bank = _compute_bank_for_category(cfg, extractor, samples)
            if cfg.memory_bank.cache_enabled:
                _save_bank_to_cache(cache_dir, category, bank)
        else:
            logger.info("命中类别特征库缓存: %s", category)
        banks[category] = bank

    global_bank = _load_bank_from_cache(cache_dir, "_GLOBAL_") if cfg.memory_bank.cache_enabled else None
    if global_bank is None:
        logger.info("构建全局特征库")
        global_bank = _compute_global_bank(cfg, extractor, train_samples)
        if cfg.memory_bank.cache_enabled:
            _save_bank_to_cache(cache_dir, "_GLOBAL_", global_bank)
    else:
        logger.info("命中全局特征库缓存")

    banks["_GLOBAL_"] = global_bank
    return banks


def _compute_bank_for_category(cfg: AppConfig, extractor: FeatureExtractor, samples: list[MultiViewSample]) -> Any:
    """从一个类别的正常样本构建 patch bank。"""

    rng = np.random.default_rng(cfg.memory_bank.random_seed)
    patches: list[Any] = []

    for sample in samples:
        for view_path in sample.view_paths:
            img = resize_image(load_rgb_image(view_path), size=cfg.dataset.image_size)
            feats = extractor.extract_patch_features(img)
            picked = _random_pick_patches(feats, rng=rng, k=cfg.memory_bank.patches_per_image)
            patches.append(picked)

    bank = _concat_patches(patches)
    if bank.shape[0] > cfg.memory_bank.max_patches_per_category:
        idx = rng.choice(bank.shape[0], size=cfg.memory_bank.max_patches_per_category, replace=False)
        bank = bank[idx]
    return bank


def _random_pick_patches(feats: PatchFeatures, *, rng: np.random.Generator, k: int) -> Any:
    """从 patch tokens 随机采样 k 个向量。"""

    patch = feats.patch_tokens.detach().cpu()
    n = int(patch.shape[0])
    if n <= k:
        return patch
    idx = rng.choice(n, size=k, replace=False)
    return patch[idx]


def _concat_patches(patches: list[Any]) -> Any:
    """拼接 patch 列表为一个二维张量。"""

    if not patches:
        raise RuntimeError("空 patch 列表")
    import torch

    return torch.cat(patches, dim=0)


def _compute_global_bank(cfg: AppConfig, extractor: FeatureExtractor, train_samples: list[MultiViewSample]) -> Any:
    """从全部训练样本构建全局兜底 patch bank（限制最大数量）。"""

    import torch

    rng = np.random.default_rng(cfg.memory_bank.random_seed)
    patches: list[Any] = []
    total = 0

    for sample in train_samples:
        for view_path in sample.view_paths:
            img = resize_image(load_rgb_image(view_path), size=cfg.dataset.image_size)
            feats = extractor.extract_patch_features(img)
            picked = _random_pick_patches(feats, rng=rng, k=cfg.memory_bank.patches_per_image)
            patches.append(picked)
            total += int(picked.shape[0])
            if total >= cfg.memory_bank.global_max_patches:
                break
        if total >= cfg.memory_bank.global_max_patches:
            break

    bank = torch.cat(patches, dim=0) if patches else torch.empty((0, 1), dtype=torch.float32)
    if bank.shape[0] == 0:
        raise RuntimeError("全局特征库构建失败：未采样到任何 patch")
    if bank.shape[0] > cfg.memory_bank.global_max_patches:
        idx = rng.choice(bank.shape[0], size=cfg.memory_bank.global_max_patches, replace=False)
        bank = bank[idx]
    return bank


def _cache_path(cache_dir: Path, category: str) -> Path:
    """返回缓存文件路径。"""

    safe = category.replace("/", "_")
    return cache_dir / f"{safe}.npz"


def _save_bank_to_cache(cache_dir: Path, category: str, bank: Any) -> None:
    """将 bank 保存为 npz。"""

    p = _cache_path(cache_dir, category)
    ensure_dir(p.parent)
    arr = bank.detach().cpu().numpy().astype(np.float16)
    np.savez_compressed(p, bank=arr)


def _load_bank_from_cache(cache_dir: Path, category: str) -> Any | None:
    """从缓存加载 bank，失败返回 None。"""

    p = _cache_path(cache_dir, category)
    if not p.exists():
        return None

    try:
        import torch

        data = np.load(p)
        arr = data["bank"].astype(np.float32)
        return torch.from_numpy(arr)
    except Exception:
        return None


def _predict_and_write_masks(
    *,
    cfg: AppConfig,
    extractor: FeatureExtractor,
    banks: dict[str, Any],
    test_samples: list[MultiViewSample],
    predicted_masks_dir: Path,
) -> list[tuple[str, float]]:
    """对测试集推理并写出 mask。"""

    rows: list[tuple[str, float]] = []
    for sample in test_samples:
        try:
            bank = banks.get(sample.category, banks["_GLOBAL_"])
            score, view_masks = _predict_one(cfg, extractor, bank, sample)
            _write_view_masks(predicted_masks_dir, sample, view_masks)
            rows.append((sample.group_folder, float(score)))
            logger.info(
                "样本推理成功: %s | category=%s | score=%.6f | views=%d",
                sample.group_folder,
                sample.category,
                float(score),
                len(sample.view_paths),
            )
        except Exception as exc:
            logger.exception("样本推理失败: %s (%s)", sample.group_folder, exc)
            zeros = [np.zeros((cfg.dataset.image_size, cfg.dataset.image_size), dtype=np.uint8) for _ in sample.view_paths]
            _write_view_masks(predicted_masks_dir, sample, zeros)
            rows.append((sample.group_folder, 0.0))
    return rows


def _predict_one(
    cfg: AppConfig,
    extractor: FeatureExtractor,
    bank: Any,
    sample: MultiViewSample,
) -> tuple[float, list[np.ndarray]]:
    """预测单个样本，返回 anomaly_score 与每视角 mask（uint8）。"""

    import torch.nn.functional as F

    view_scores: list[float] = []
    view_masks: list[np.ndarray] = []
    bank_t = None

    for view_path in sample.view_paths:
        img = resize_image(load_rgb_image(view_path), size=cfg.dataset.image_size)
        feats = extractor.extract_patch_features(img)
        patch = feats.patch_tokens
        if bank_t is None:
            bank_t = bank.to(patch.device)

        max_sim = max_cosine_similarity(patch, bank_t, chunk_size=cfg.inference.nn_chunk_size)
        dist = (1.0 - max_sim).clamp(min=0.0)

        h, w = feats.grid_size
        patch_map = dist.reshape(1, 1, h, w)
        up = F.interpolate(patch_map, size=(cfg.dataset.image_size, cfg.dataset.image_size), mode="bilinear", align_corners=False)
        m = up.squeeze().detach().cpu().numpy().astype(np.float32)

        score = float(m.max())
        view_scores.append(score)
        view_masks.append(_to_uint8_mask(m))

    agg = cfg.inference.score_aggregation.lower().strip()
    if agg == "mean":
        return float(np.mean(view_scores)), view_masks
    return float(np.max(view_scores)), view_masks


def _to_uint8_mask(m: np.ndarray) -> np.ndarray:
    """将 float32 热力图归一化为 uint8。"""

    vmin = float(np.min(m))
    vmax = float(np.max(m))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(m, dtype=np.uint8)
    x = (m - vmin) / (vmax - vmin)
    x = (x * 255.0).clip(0.0, 255.0)
    return x.astype(np.uint8)


def _write_view_masks(predicted_masks_dir: Path, sample: MultiViewSample, view_masks: list[np.ndarray]) -> None:
    """写出 5 视角 mask 文件。"""

    out_dir = predicted_masks_dir / sample.category / sample.sample_id
    ensure_dir(out_dir)
    for i, mask in enumerate(view_masks):
        save_grayscale_png(out_dir / f"{i}_mask.png", array_2d_uint8=mask)
