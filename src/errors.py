"""运行指南

该模块定义应用错误类型，供上层捕获并做容错处理。
通常不直接执行。
"""

from __future__ import annotations


class ComplexAbnormalDetectionError(Exception):
    """应用级通用异常。"""


class ConfigError(ComplexAbnormalDetectionError):
    """配置加载或校验失败异常。"""


class DependencyMissingError(ComplexAbnormalDetectionError):
    """关键依赖缺失异常。"""


class DatasetStructureError(ComplexAbnormalDetectionError):
    """数据集目录结构不符合预期异常。"""

