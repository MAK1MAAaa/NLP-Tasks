# Task-1：基于机器学习的文本分类

## 任务要求
- 学习《神经网络与深度学习》第三章，重点关注线性模型与 Softmax 回归。
- **数据处理**: 使用 `pandas` 读取 `new_train.tsv` 和 `new_test.tsv`。
- **特征提取**: 实现 **Bag of Words (BoW)** 和 **N-gram** 特征提取。
- **模型实现**: 手动实现 Softmax 回归，**禁止直接调用 `torch.nn` 中的高阶函数**（如 `nn.Linear`, `nn.CrossEntropyLoss`）。
- **实验对比**: 测试不同特征提取方式、学习率对性能的影响，并绘制训练曲线。

## 实验环境
- **Python**: 3.10+
- **依赖管理**: `uv` (根目录下 `pyproject.toml`)
- **主要库**: `torch` (仅用于矩阵运算), `pandas`, `scikit-learn`, `matplotlib`

## 运行指南

### 1. 环境准备
在项目根目录下，使用 `uv` 同步依赖：
```bash
uv sync
```

### 2. 数据准备
确保 `Task1/data/` 目录下包含以下文件：
- `new_train.tsv`: 训练数据集 (文本 \t 标签)
- `new_test.tsv`: 测试数据集 (文本 \t 标签)

### 3. 运行训练与实验
执行训练脚本，该脚本会自动运行 BoW 和 N-gram 的对比实验：
```bash
uv run python Task1/code/train.py
```

## 实验结果与分析

### 1. 特征提取对比 (BoW vs. N-gram)

| 特征提取方式 | 词表大小 (Max Features) | 验证集最高准确率 | 测试集准确率 |
| :--- | :--- | :--- | :--- |
| **Bag of Words (Unigram)** | 2000 | **0.4549** | **0.4639** |
| **N-gram (Bigram)** | 2000 | 0.3875 | 0.4062 |

> **分析**: 在本次实验中，**Unigram (BoW)** 的表现显著优于 **Bigram**。这主要是因为在词表大小限制为 2000 的情况下，Unigram 能够覆盖绝大多数高频核心词汇，而 Bigram 特征极其稀疏，2000 个特征不足以捕捉到足够的有效组合，导致模型欠拟合。

### 2. 训练曲线展示
训练过程中的 Loss 下降情况与 Accuracy 变化曲线已保存至 `Task1/result/experiment_results.png`。

- **Loss 变化**: 随着 Epoch 增加，Unigram 的 Loss 下降速度明显快于 Bigram，最终收敛值也更低。
- **准确率变化**: Unigram 在验证集上的表现一直保持领先，且在 50 个 Epoch 内未出现明显的过拟合迹象。

### 3. 核心实现说明 (Softmax Regression)

为了符合“不调用 `torch.nn` 函数”的要求，核心逻辑实现如下：

- **前向传播**: `logits = X @ W + b`，随后通过手动实现的 `softmax` 函数转化为概率分布。
- **损失函数**: 手动实现交叉熵 `loss = -mean(log(probs_target))`。
- **参数更新**: 利用 `loss.backward()` 计算梯度后，通过 `with torch.no_grad(): W -= lr * W.grad` 手动更新权重并清空梯度。

## 常见问题排查

1. **FileNotFoundError**:
   - 请检查 `Task1/data/` 目录下是否存在 `.tsv` 文件。
2. **RuntimeError (Trying to backward through the graph a second time)**:
   - 确保在参数更新后及时调用 `grad.zero_()`，且参数初始化为 Leaf Tensor。
3. **Matplotlib 绘图问题**:
   - 脚本会自动将图片保存至 `Task1/result/` 目录。
