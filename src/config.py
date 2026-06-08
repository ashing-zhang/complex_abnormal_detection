"""运行指南

该模块包含配置数据模型（配置与代码分离）。
通常不直接执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetConfig:
    """数据集路径与基础参数。"""

    train_dir: Path
    test_dir: Path
    view_count: int = 5
    image_size: int = 448


@dataclass(frozen=True, slots=True, kw_only=True)
class DINOv2Config:
    """DINOv2 加载配置。"""

    repo_or_dir: str = "facebookresearch/dinov2"
    model_name: str = "dinov2_vits14"
    trust_repo: bool | str = True
    force_reload: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class FeatureExtractorConfig:
    """特征提取器配置。"""

    backend: str = "dinov2"
    device: str = "auto"
    dinov2: DINOv2Config = DINOv2Config()


@dataclass(frozen=True, slots=True, kw_only=True)
class MemoryBankConfig:
    """内存库/特征库构建配置。"""

    patches_per_image: int = 128
    max_patches_per_category: int = 200_000
    global_max_patches: int = 200_000
    random_seed: int = 2026
    cache_enabled: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceConfig:
    """推理阶段配置。"""

    nn_chunk_size: int = 4096
    score_aggregation: str = "max"


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputConfig:
    """输出路径配置。"""

    work_dir: Path
    zip_name: str = "submission.zip"
    submission_csv_name: str = "submission.csv"
    predicted_masks_dir_name: str = "predicted_masks"


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingConfig:
    """日志配置。"""

    level: str = "INFO"


@dataclass(frozen=True, slots=True, kw_only=True)
class AppConfig:
    """应用配置根对象。"""

    dataset: DatasetConfig
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    memory_bank: MemoryBankConfig = MemoryBankConfig()
    inference: InferenceConfig = InferenceConfig()
    output: OutputConfig = OutputConfig(work_dir=Path("outputs"))
    logging: LoggingConfig = LoggingConfig()

    @staticmethod
    def from_dict(data: dict[str, Any], *, base_dir: Path) -> "AppConfig":
        """从字典构建 AppConfig，并将相对路径按 base_dir 解析。"""

        dataset = DatasetConfig(
            train_dir=(base_dir / Path(_required(data, "dataset.train_dir"))).resolve(),
            test_dir=(base_dir / Path(_required(data, "dataset.test_dir"))).resolve(),
            view_count=int(_optional(data, "dataset.view_count", 5)),
            image_size=int(_optional(data, "dataset.image_size", 448)),
        )

        dinov2_data = _optional(data, "feature_extractor.dinov2", {}) or {}
        dinov2 = DINOv2Config(
            repo_or_dir=str(_optional(dinov2_data, "repo_or_dir", "facebookresearch/dinov2")),
            model_name=str(_optional(dinov2_data, "model_name", "dinov2_vits14")),
            trust_repo=_optional(dinov2_data, "trust_repo", True),
            force_reload=bool(_optional(dinov2_data, "force_reload", False)),
        )

        feature_extractor = FeatureExtractorConfig(
            backend=str(_optional(data, "feature_extractor.backend", "dinov2")),
            device=str(_optional(data, "feature_extractor.device", "auto")),
            dinov2=dinov2,
        )

        memory_bank = MemoryBankConfig(
            patches_per_image=int(_optional(data, "memory_bank.patches_per_image", 128)),
            max_patches_per_category=int(_optional(data, "memory_bank.max_patches_per_category", 200_000)),
            global_max_patches=int(_optional(data, "memory_bank.global_max_patches", 200_000)),
            random_seed=int(_optional(data, "memory_bank.random_seed", 2026)),
            cache_enabled=bool(_optional(data, "memory_bank.cache_enabled", True)),
        )

        inference = InferenceConfig(
            nn_chunk_size=int(_optional(data, "inference.nn_chunk_size", 4096)),
            score_aggregation=str(_optional(data, "inference.score_aggregation", "max")),
        )

        output = OutputConfig(
            work_dir=(base_dir / Path(_optional(data, "output.work_dir", "outputs"))).resolve(),
            zip_name=str(_optional(data, "output.zip_name", "submission.zip")),
            submission_csv_name=str(_optional(data, "output.submission_csv_name", "submission.csv")),
            predicted_masks_dir_name=str(_optional(data, "output.predicted_masks_dir_name", "predicted_masks")),
        )

        logging_cfg = LoggingConfig(level=str(_optional(data, "logging.level", "INFO")))

        return AppConfig(
            dataset=dataset,
            feature_extractor=feature_extractor,
            memory_bank=memory_bank,
            inference=inference,
            output=output,
            logging=logging_cfg,
        )


def _required(data: dict[str, Any], dotted_key: str) -> Any:
    """读取必填字段（支持 a.b.c 形式）。"""

    value = _optional(data, dotted_key, None)
    if value is None:
        raise KeyError(dotted_key)
    return value


def _optional(data: dict[str, Any], dotted_key: str, default: Any) -> Any:
    """读取可选字段（支持 a.b.c 形式）。"""

    current: Any = data
    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current

