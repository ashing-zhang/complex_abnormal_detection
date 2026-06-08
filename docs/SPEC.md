**【核心任务】**

聚焦复杂工业场景下的多视角异常检测。需设计算法模型，根据工业相机阵列从 5 个视角拍摄的零部件图像，完成以下任务：

1. 图像级异常分类：判断该零件样本是否存在缺陷，并输出异常概率得分。
2. 像素级异常分割：对划痕、凹陷、破损、污渍等微小缺陷区域进行像素级定位，输出异常热力图或缺陷 mask。
3. 未见类别泛化：模型需具备较强的零样本泛化能力，能够应对新产品冷启动场景下完全未见过的工业类别。

**【数据集介绍】**

使用 Real-IAD Variety 真实工业多视角异常检测数据集的深度清洗子集。

数据规模：包含 100 个不同工业零部件类别，共计 3.1 万余个文件。

数据划分：

Train（训练集）：
仅包含 50 个已见类别的正常样本，供参赛选手进行无监督或弱监督异常检测建模。

Test A（A 榜测试集）：
包含训练集中 50 个已见类别的正常与异常样本，用于评估模型在已见类别上的异常分类与像素级分割能力。

物理结构：
每个样本以自包含的 Sxxxx 文件夹形式呈现，文件夹内部保留 5 个相机视角的空间拓扑顺序，图像命名为 0.png 至 4.png。

**【生成结果说明】**

需生成 .zip 格式的预测结果压缩包。压缩包解压后必须包含以下内容：

**1.submission.csv**

submission.csv 必须包含以下两列：

group_folder：
样本路径，例如：

3_adapter/S0001

anomaly_score：
图像级异常概率得分。数值越大，表示该样本越可能存在异常。

submission.csv 示例：

group_folder,anomaly_score
3_adapter/S0001,0.0321
3_adapter/S0002,0.8765
battery/S0001,0.1248

**2.predicted_masks/ 文件夹**

predicted_masks/ 用于存放像素级异常预测热力图。其目录结构需与样本路径严格对应。

示例结构：

- predicted_masks/
  - 3_adapter/
    - S0001/
      - 0_mask.png
      - 1_mask.png
      - 2_mask.png
      - 3_mask.png
      - 4_mask.png
    - S0002/
      - 0_mask.png
      - 1_mask.png
      - 2_mask.png
      - 3_mask.png
      - 4_mask.png

每个样本包含 5 个视角，对应文件命名为：

0_mask.png
1_mask.png
2_mask.png
3_mask.png
4_mask.png

mask 要求：

1. mask 图像应为单通道灰度图。
2. mask 尺寸应为 448x448。
3. 像素值越大，表示该位置越可能为异常区域。
4. 无异常视角可生成全黑 mask。

最终生成 zip 解压后的推荐结构如下：

- submission.csv
- predicted_masks/
  - 类别名/
    - 样本文件夹/
      - 0_mask.png
      - 1_mask.png
      - 2_mask.png
      - 3_mask.png
      - 4_mask.png