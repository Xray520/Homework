# 医学影像伪影鲁棒性：轻量复现

此项目复现论文的评测流程：在医学影像中加入强、弱伪影，观察分类模型在原始与退化图像上的准确率、敏感度和相对性能变化。默认用固定随机种子生成离线合成胸片样式图像，无需下载即可运行；联网后可切换到官方 PneumoniaMNIST。

## 与原论文的差别

原论文评估多个视觉语言模型（VLM），覆盖 MRI、OCT、胸部 X 光、五类伪影和真实伪影。本项目为快速演示提供单模态、28×28、小样本的轻量流程。

默认数据 SyntheticXrayLike 是程序生成的肺野、肋骨纹理和局部阴影图案，不是患者影像或真实医学数据集。项目复现的是伪影生成与分层鲁棒性评估流程，不是原论文模型或其数值结果的逐项复现。

## 实验内容

- 默认数据：SyntheticXrayLike，二分类 28×28 合成灰度图，仅用于验证代码管线。
- 可选数据：PneumoniaMNIST，二分类儿科胸部 X 光，官方 28×28 数据集；首次运行时从 Zenodo 下载。
- 模型：scikit-learn MLPClassifier，CPU 可运行。
- 伪影：Gaussian noise、bias field、motion、random crop、rotation，各含 weak/strong 两档。
- 指标：accuracy、pneumonia sensitivity、相对 clean accuracy 变化、低置信度比例。
- 随机种子默认 42，便于重复运行。

## 运行

需要 Python 3.10 或更新版本：

    python -m pip install -r requirements.txt
    python src/run_reproduction.py

默认命令离线可跑。使用官方 PneumoniaMNIST：

    python src/run_reproduction.py --dataset medmnist

可调小样本和迭代次数：

    python src/run_reproduction.py --train-limit 1200 --test-limit 468 --max-iter 18 --seed 42

MedMNIST 模式首次运行会下载约 4.2 MB 数据到 data/。结果写入 results/：

- metrics.csv：clean、各伪影类型和严重程度的指标。
- summary.json：运行设置和 clean 基线。
- results_screenshot.png：结果图。
- model.joblib：训练后的轻量分类模型。

## 指标说明

- performance_change_pct = (伪影准确率 − clean 准确率) / clean 准确率 × 100。
- low_confidence_rate 使用 clean 验证集置信度最低 5% 分位作为阈值，是质量警示代理，不等同于论文中 VLM 自行报告图像质量差的检测率。
- 每张图的伪影参数由固定种子生成。

## 原论文对照与限制

原研究提出分类表现、弱伪影下的相对性能变化、强伪影下的质量问题识别三层评估思路，并覆盖 MRI、OCT 和 X 光。默认合成数据只验证评测管线能运行，不能与论文表格作数值等价比较。切换到 --dataset medmnist 可进行真实胸片数据上的简化实验，但仍与原研究的 VLM、多个模态和人工/自报告指标不同。

## 来源

- MedMNIST 官方项目：https://medmnist.com/
- PneumoniaMNIST Zenodo 数据：https://zenodo.org/records/10519652
- 数据集许可证：MedMNIST 官方信息列明 PneumoniaMNIST 为 CC BY 4.0。
- 原论文：Understanding the robustness of vision-language models to medical image artefacts，https://doi.org/10.1038/s41746-025-02108-w
- 原论文代码与基准：https://github.com/ziijiecheng/VLM_robustness

仅用于教学和研究方法演示，不用于临床决策。