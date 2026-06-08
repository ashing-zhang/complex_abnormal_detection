"""运行指南

从项目根目录执行：
python -m src.predict

本模块负责加载 yaml 配置并转换为配置对象。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import AppConfig
from .errors import ConfigError, DependencyMissingError


def load_app_config(config_path: Path) -> AppConfig:
    """从 yaml 文件加载 AppConfig。"""

    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")

    raw = _read_yaml(config_path)
    if not isinstance(raw, dict):
        raise ConfigError("配置根节点必须为 dict")

    try:
        return AppConfig.from_dict(raw, base_dir=config_path.parent)
    except KeyError as exc:
        raise ConfigError(f"缺少必填配置项: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"配置解析失败: {exc}") from exc


def default_config_path() -> Path:
    """返回内置默认配置文件路径。"""

    return (Path(__file__).resolve().parent / "configs" / "default.yaml").resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 yaml 为 dict。"""

    try:
        import yaml
    except Exception as exc:
        raise DependencyMissingError("缺少依赖 PyYAML，请先安装: pip install pyyaml") from exc

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigError(f"读取 yaml 失败: {path}") from exc

    if not isinstance(data, dict):
        raise ConfigError("yaml 顶层必须为 dict")

    return data

