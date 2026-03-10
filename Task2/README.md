# Task-2：基于深度学习的文本分类

## 任务要求
- 学习卷积神经网络 (CNN)、循环神经网络 (RNN) 及 Transformer 在文本分类中的应用。
- **数据处理**: 使用 `pandas` 读取 `new_train.tsv` 和 `new_test.tsv`。
- **特征提取**: 使用 `torch.nn.Embedding` 将文本转换为向量序列。
- **模型实现**: 
  - 实现 **TextCNN** (参考论文 *Convolutional Neural Networks for Sentence Classification*)。
  - 实现 **Bi-LSTM** (双向长短期记忆网络)。
  - 实现 **Transformer Encoder** (基于注意力机制的模型)。
  - 支持使用 **GloVe** 预训练词向量进行初始化。
- **实验对比**: 测试不同模型结构、初始化方式对最终分类性能的影响。

## 实验环境
- **Python**: 3.10+
- **依赖管理**: `uv` (根目录下 `pyproject.toml`)
- **主要库**: `torch` (使用 GPU/CUDA), `pandas`, `scikit-learn`, `matplotlib`

## 运行指南

### 1. 环境准备
在项目根目录下，使用 `uv` 同步依赖：
```bash
uv sync
```

### 2. 数据准备
确保 `Task2/data/` 目录下包含以下文件：
- `new_train.tsv`: 训练数据集
- `new_test.tsv`: 测试数据集
- `glove/glove.6B.100d.txt`: GloVe 预训练词向量

### 3. 运行训练与实验
执行训练脚本，该脚本会自动运行 CNN、RNN、Transformer 及消融实验：
```bash
uv run python Task2/code/train.py
```

## 实验结果与分析

### 1. 模型性能对比 (全量实验)

| 模型 | 词向量初始化 | 验证集最高准确率 | 测试集准确率 |
| :--- | :--- | :--- | :--- |
| **TextCNN** | **GloVe (100d)** | 0.4994 | 0.5095 |
| **Bi-LSTM** | **GloVe (100d)** | **0.5246** | **0.5134** |
| **Transformer** | **GloVe (100d)** | 0.4936 | 0.4548 |
| **TextCNN** | Random | 0.4484 | 0.4300 |

> **分析**: 
> 1. **GloVe 的显著作用**: 对比 `TextCNN (GloVe)` 与 `TextCNN (Random)`，使用预训练词向量使测试集准确率提升了约 **8%** (0.5095 vs 0.4300)，且收敛速度大幅加快。
> 2. **模型结构对比**: 在本数据集中，**Bi-LSTM** 表现最稳健，取得了最高的测试集准确率 (0.5134)。**TextCNN** 紧随其后，且训练速度最快。
> 3. **Transformer 的局限性**: Transformer 在本任务中表现欠佳 (0.4548)，且出现了严重的过拟合。这主要是因为 Transformer 结构复杂，在小规模数据集上极易过拟合，且对超参数（如学习率、层数、Head 数）非常敏感。
> 4. **过拟合观察**: 所有模型在 15 个 Epoch 内均出现了不同程度的过拟合（训练 Loss 接近 0，但验证集准确率在第 5-10 轮后开始停滞或下降）。

### 2. 训练曲线展示
全量对比实验的 Loss 下降情况与 Accuracy 变化曲线已保存至 `Task2/result/full_comparison.png`。

### 3. 核心实现说明

- **TextCNN**: 使用 3, 4, 5 三种尺寸的卷积核捕捉局部特征，配合 Global Max Pooling。
- **Bi-LSTM**: 使用双向 LSTM 捕捉长距离上下文依赖，取最后时刻的隐藏状态拼接后分类。
- **Transformer**: 包含 Positional Encoding 和多层 Encoder 层，使用 Global Average Pooling 提取特征。

## 常见问题排查

1. **CUDA Out of Memory**:
   - 尝试减小 `batch_size` 或 `max_len`。
2. **Transformer 效果差**:
   - 尝试降低学习率 (如 `1e-4`) 或增加数据增强。
3. **GloVe 加载缓慢**:
   - 首次加载较大，后续运行会受磁盘 IO 限制。
