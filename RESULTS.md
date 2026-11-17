# 本次运行结果

- 数据：SyntheticXrayLike（程序生成的 28×28 合成图像，不是临床/患者 X 光）
- 随机种子：42
- 样本：训练 1,200，验证 400，测试 468
- 模型：单隐藏层 MLPClassifier，早停后训练 15 轮
- clean accuracy：90.2%
- clean pneumonia sensitivity：87.6%

## 观测到的伪影影响

| 伪影 | 弱准确率变化 | 强准确率变化 |
|---|---:|---:|
| Gaussian noise | +0.7% | -6.9% |
| Bias field | -10.0% | -34.6% |
| Motion | +0.7% | -28.2% |
| Random crop | -44.1% | -41.7% |
| Rotation | -13.5% | -35.5% |

准确率变化按相对 clean accuracy 计算。随机伪影参数会造成波动，因此弱/强的变化不一定严格单调；Random crop 的弱档本次比强档损失略大。请以 metrics.csv 的完整指标为准。

## 结果解释与限制

这次结果验证了评测流程能对不同伪影计算分类表现、相对变化和低置信度比例。它不是原论文的 VLM 数值复现：当前截图基于合成样式数据和轻量 MLP。运行 python src/run_reproduction.py --dataset medmnist 可在可访问 Zenodo 的环境中切换到 PneumoniaMNIST，再做真实胸片上的简化评估。