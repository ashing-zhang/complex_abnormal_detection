"""运行指南

该模块封装特征提取（默认使用 DINOv2）。
通常不直接执行。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .config import FeatureExtractorConfig
from .errors import DependencyMissingError


class FeatureExtractor(Protocol):
    """特征提取器协议。"""

    def extract_patch_features(self, image_rgb) -> "PatchFeatures":
        """从单张 RGB 图像提取 patch 特征与 CLS 特征。"""


@dataclass(frozen=True, slots=True)
class PatchFeatures:
    """单张图像的特征输出。"""

    patch_tokens: Any
    cls_token: Any
    grid_size: tuple[int, int]


def build_feature_extractor(cfg: FeatureExtractorConfig) -> FeatureExtractor:
    """根据配置构建特征提取器实例。"""

    backend = cfg.backend.lower().strip()
    if backend == "dinov2":
        return DINOv2FeatureExtractor.create(cfg)
    raise ValueError(f"不支持的特征提取后端: {cfg.backend}")


@dataclass(slots=True, init=False)
class DINOv2FeatureExtractor:
    """基于 torch.hub 的 DINOv2 特征提取器。"""

    _cfg: FeatureExtractorConfig
    _torch: Any
    _device: Any
    _model: Any

    @classmethod
    def create(cls, cfg: FeatureExtractorConfig) -> "DINOv2FeatureExtractor":
        """构建并初始化 DINOv2FeatureExtractor。"""

        inst = cls.__new__(cls)
        inst._cfg = cfg
        try:
            import torch
        except Exception as exc:
            raise DependencyMissingError("缺少依赖 torch，请先安装: pip install torch") from exc

        inst._torch = torch
        inst._device = _resolve_device(torch, cfg.device)
        inst._model = _load_dinov2_model(torch, cfg, device=inst._device)
        inst._model.eval()
        return inst

    def extract_patch_features(self, image_rgb) -> PatchFeatures:
        """从 PIL.Image 提取 patch tokens / cls token。"""

        torch = self._torch
        x = _pil_to_normalized_tensor(torch, image_rgb, device=self._device)
        with torch.no_grad():
            if hasattr(self._model, "forward_features"):
                out = self._model.forward_features(x)
                patch = out["x_norm_patchtokens"]
                cls = out["x_norm_clstoken"]
            else:
                patch, cls = _forward_vit_fallback(self._model, x)

        patch = _l2_normalize(patch, dim=-1)
        cls = _l2_normalize(cls, dim=-1)

        grid = _infer_grid(patch)
        return PatchFeatures(
            patch_tokens=patch.squeeze(0),
            cls_token=cls.squeeze(0),
            grid_size=grid,
        )


def _load_dinov2_model(torch: Any, cfg: FeatureExtractorConfig, *, device: Any) -> Any:
    """通过 torch.hub.load 加载 DINOv2 模型。"""

    dcfg = cfg.dinov2
    try:
        model = torch.hub.load(
            dcfg.repo_or_dir,
            dcfg.model_name,
            trust_repo=dcfg.trust_repo,
            force_reload=dcfg.force_reload,
        )
    except Exception as exc:
        raise DependencyMissingError(
            "加载 DINOv2 失败：请确认网络可用或 repo_or_dir/model_name 配置正确"
        ) from exc

    return model.to(device)


def _resolve_device(torch, device: str) -> "torch.device":
    """解析 device 字符串。"""

    d = device.lower().strip()
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def _pil_to_normalized_tensor(torch, image_rgb, *, device):
    """将 PIL.Image 转为归一化 Tensor[B,3,H,W]。"""

    try:
        import numpy as np
    except Exception as exc:
        raise DependencyMissingError("缺少依赖 numpy，请先安装: pip install numpy") from exc

    arr = np.asarray(image_rgb, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("输入图像必须为 RGB")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    chw = arr.transpose(2, 0, 1)
    x = torch.from_numpy(chw).unsqueeze(0).to(device=device)
    return x


def _forward_vit_fallback(model, x):
    """兼容没有 forward_features 的 ViT：尝试获取中间层输出。"""

    if not hasattr(model, "get_intermediate_layers"):
        raise DependencyMissingError("该模型不支持 patch 特征提取（缺少 get_intermediate_layers）")

    out = model.get_intermediate_layers(x, n=1, return_class_token=True)
    last = out[0]
    if isinstance(last, tuple) and len(last) == 2:
        patch, cls = last
        return patch, cls

    raise DependencyMissingError("无法从 get_intermediate_layers 输出解析 patch/cls tokens")


def _l2_normalize(x, *, dim: int):
    """对最后一维做 L2 归一化。"""

    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)


def _infer_grid(patch_tokens) -> tuple[int, int]:
    """根据 patch token 数推断网格大小。"""

    n = patch_tokens.shape[1]
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"无法推断 patch 网格: tokens={n}")
    return (side, side)

