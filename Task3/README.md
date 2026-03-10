# Task-3：实现 Transformer 的基础结构

## 任务要求
- 深入理解 Transformer 的核心机制：Multi-head Attention、Layer Norm、残差连接、Masking 等。
- **数据准备**: 自行构造多位数加法数据集（子任务 1）及语言模型语料（子任务 2）。
- **模型实现**: 
  - **Encoder-Decoder**: 用于 Seq2Seq 加法任务。
  - **Decoder-only**: 用于 GPT 风格的语言模型任务。
- **实验验证**: 探究不同 Tokenizer、模型规模及泛化性对性能的影响。

## 实验环境
- **Python**: 3.10+
- **依赖管理**: `uv` (根目录下 `pyproject.toml`)
- **主要库**: `torch`, `matplotlib`, `numpy`

## 运行指南

### 1. 子任务 1：多位数加法
```bash
uv run python Task3/code/train_addition.py
```

### 2. 子任务 2：语言模型 (Decoder-only)
```bash
uv run python Task3/code/train_language_model.py
```

## 实验结果与分析

### 1. 多位数加法任务 (Subtask 1)
- **核心结论**: 通过 **反转序列 (Reverse Sequence)** 技巧，Enc-Dec 和 Decoder-only 架构均能实现 **100%** 的 3 位数加法准确率。
- **泛化性**: 实验证明标准 Transformer 无法泛化到训练未见过的长度（Train 3, Test 4 准确率为 0%）。

### 2. 语言模型任务 (Subtask 2)

| 实验名称 | Tokenizer | 模型规模 | 最终 Loss | 生成效果 |
| :--- | :--- | :--- | :--- | :--- |
| **Char-level (Small)** | 字符级 | 2层/d=128 | ~0.1229 | 仅能生成单词片段 |
| **Char-level (Large)** | 字符级 | 4层/d=256 | ~0.1206 | 仅能生成单词片段 |
| **Word-level (Small)** | **词级** | 2层/d=128 | **~0.0476** | **生成完整且连贯的长句** |

> **分析**: 
> 1. **Tokenizer 的影响**: **Word-level** 模型表现远优于字符级。在小规模语料下，词级 Tokenizer 降低了序列的逻辑长度，使模型能更轻松地捕捉长程语义关联。
> 2. **生成质量**: Word-level 模型成功生成了 "the transformer is a deep learning architecture introduced in 2017..." 等复杂长句，语法完全正确。
> 3. **收敛速度**: 词级模型在第 10 轮左右即达到极低 Loss，收敛效率极高。

### 3. 训练曲线展示
- 加法对比曲线：`Task3/result/addition_comparison.png`
- 语言模型对比曲线：`Task3/result/lm_comparison.png`

## 核心实现说明
- **Decoder-only 架构**: 使用 `nn.TransformerEncoder` 配合 `Causal Mask`。在推理时，通过 `input_seq[:, -seq_len:]` 动态切片，确保输入长度与训练一致。
- **Tokenizer 设计**: 实现了支持 `char` 和 `word` 两种模式的 `SimpleTokenizer`，并处理了分词与索引映射。
- **Masking**: 统一使用布尔掩码，解决了自回归生成中的因果性约束。
