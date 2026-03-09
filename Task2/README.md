Task-2：基于深度学习的文本分类
原始任务：https://github.com/FudanNLP/nlp-beginner task-2
修订：@用户8817 @用户2245 
前置条件
- 学习《神经网络与深度学习》至第六章，重点关注第五章 “卷积神经网络” 、第六章 “循环神经网络” 部分
- 阅读论文 Convolutional Neural Networks for Sentence Classification https://arxiv.org/abs/1408.5882
- 学习 word embedding、dropout 的基本思想及操作

代码实现阶段

0.准备阶段
- 学习 pytorch 库的基本操作，重点关注embedding 及 CNN 的部分，及使用 GPU （cuda）的基本操作。
- Task 2不要求实现太多具体的东西，核心部分都是调库。

1.数据的下载与读取
- 同 Task-1 。

2.数据的预处理与划分
- 数据集的划分同 同 Task-1 。
- 将句子转换为序列后直接调用 torch.nn.embedding 即可完成 embedding 操作。

3.模型的训练
- 调用 pytorch 关于 CNN 的库函数即可完成。
- 注意应将模型大多数操作转移到 GPU 上运行提升训练速度。

实验阶段

- 测试不同的损失函数、学习率对最终分类性能的影响
- 测试卷积核个数、大小及不同的优化器对最终分类性能的影响
- 测试使用 glove 预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/ 对最终分类性能的影响
- 测试 CNN 改为 RNN 、Transformer （直接调用 pytorch 中的 api）等其它模型对最终分类性能的影响
- 将结果绘制成图表

