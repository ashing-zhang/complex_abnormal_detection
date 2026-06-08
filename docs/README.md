# Complex Abnormal Detection

本项目实现一个面向 Real-IAD Variety 多视角工业异常检测任务的可运行基线方案，按竞赛要求生成 `submission.csv` 与像素级 `predicted_masks/`，并打包为 `submission.zip`。

## 任务说明

任务目标与提交格式以 [SPEC.md](file:///d:/codes/complex_abnormal_detection/docs/SPEC.md) 为准，核心包括：

- 图像级异常分类：输出样本级 `anomaly_score`
- 像素级异常分割：输出每个样本 5 个视角的异常热力图（mask）
- 未见类别泛化：仅使用训练集正常样本建模，期望具备零样本泛化能力

## 数据组织

数据集在本仓库中约定为：

- `data/Train/{category}/Sxxxx/{0..4}.png`：训练集，仅正常样本
- `data/Test_A/{category}/Sxxxx/{0..4}.png`：测试集

每个样本由 5 个视角组成，文件名为 `0.png` 至 `4.png`。

## 方案概览（核心逻辑）

实现以“基础模型特征 + 正常特征库（Memory Bank）+ 最近邻距离”作为无监督异常检测基线：

1. 使用 DINOv2 提取每张视角图像的 patch 特征（L2 归一化）
2. 从训练集中正常样本随机采样 patch 特征，构建类别级 Memory Bank，并额外构建一个全局兜底特征库
3. 推理时将测试 patch 特征与对应类别的 Memory Bank 做余弦相似度最近邻匹配：
   - `anomaly_map = 1 - max_cosine_similarity(patch, bank)`
4. 将 patch 网格热力图上采样到 `448x448` 得到像素级 mask（灰度值越大越异常）
5. 多视角融合默认取 5 个视角的最大异常分数作为 `anomaly_score`（可配置为 mean）

参考建议与背景可见 [HELP.md](file:///d:/codes/complex_abnormal_detection/docs/HELP.md)。

## 运行方式

从项目根目录执行：

```bash
python -m src.predict
```

配置方式（推荐）：

- 默认配置文件：`src/configs/default.yaml`
- 或通过环境变量指定配置文件路径：`COMPLEX_ABNORMAL_DETECTION_CONFIG=/path/to/your.yaml`

## 输出结构

默认输出到 `outputs/`（可在配置中修改）：

- `outputs/submission.csv`
- `outputs/predicted_masks/{category}/{Sxxxx}/{0..4}_mask.png`
- `outputs/submission.zip`

mask 约束：

- 单通道灰度图
- 尺寸 `448x448`
- 像素值越大表示越可能为异常区域

## 代码结构

代码位于 `src/`，保持模块化与配置驱动：

- [predict.py](file:///d:/codes/complex_abnormal_detection/src/predict.py)：入口（读取配置、初始化日志、运行流水线）
- [config.py](file:///d:/codes/complex_abnormal_detection/src/config.py)：配置数据模型
- [config_loader.py](file:///d:/codes/complex_abnormal_detection/src/config_loader.py)：yaml 配置加载
- [services.py](file:///d:/codes/complex_abnormal_detection/src/services.py)：核心用例编排（构建特征库、推理、写出提交）
- [feature_extractor.py](file:///d:/codes/complex_abnormal_detection/src/feature_extractor.py)：DINOv2 特征提取
- [dataset.py](file:///d:/codes/complex_abnormal_detection/src/dataset.py)：数据集扫描与样本结构
- [nn.py](file:///d:/codes/complex_abnormal_detection/src/nn.py)：分块最近邻相似度计算
- [io.py](file:///d:/codes/complex_abnormal_detection/src/io.py)：写 csv、mask 与 zip

## 依赖说明

运行时依赖（需自行安装）：

- `torch`
- `numpy`
- `pillow`
- `pyyaml`

