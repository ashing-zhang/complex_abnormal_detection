"""运行指南

从项目根目录运行：
python -m src.predict

自定义配置：
- 默认读取内置配置：src/configs/default.yaml
- 或设置环境变量 COMPLEX_ABNORMAL_DETECTION_CONFIG 指向你的 yaml 配置文件

输出：
- outputs/submission.csv
- outputs/predicted_masks/...
- outputs/submission.zip
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config_loader import default_config_path, load_app_config
from .errors import ComplexAbnormalDetectionError
from .services import run_prediction_pipeline


def main() -> None:
    """程序入口。"""

    config_path = _resolve_config_path()
    cfg = load_app_config(config_path)

    logging.basicConfig(
        level=cfg.logging.level.upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    logger = logging.getLogger("complex_abnormal_detection")
    logger.info("配置文件: %s", config_path)

    try:
        paths = run_prediction_pipeline(cfg)
    except ComplexAbnormalDetectionError:
        logger.exception("任务失败")
        raise

    logger.info("submission.csv: %s", paths.submission_csv)
    logger.info("predicted_masks/: %s", paths.predicted_masks_dir)
    logger.info("submission.zip: %s", paths.zip_path)


def _resolve_config_path() -> Path:
    """解析配置文件路径。"""

    env = os.getenv("COMPLEX_ABNORMAL_DETECTION_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return default_config_path()


if __name__ == "__main__":
    main()

