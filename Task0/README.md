# Task-0：熟悉 WSL 与 vLLM 操作

## 任务要求
- 使用一张 3090/4090 显卡，使用 vllm 框架部署一个模型的 API 
- 比较使用 vllm 和直接使用 transfromers 进行推理的效率，并思考其中的原因
- **本地 WSL 运行**: 在 WSL2 (Ubuntu 22.04) 环境下使用 `uv` 管理依赖。
- **数据持久化**: 模型保存在 `Task0/data` 目录下。

## 实验环境
- GPU: NVIDIA GeForce RTX 3060/4070 (12GB) 或更高
- Model: **Qwen/Qwen3-0.6B** (轻量级模型，适合本地测试)
- Frameworks: vLLM, Transformers, PyTorch, **uv**

## 运行指南 (本地 WSL)

### 1. 环境准备
在 WSL2 终端进入 `Task0/code/` 目录，执行初始化脚本：
```bash
bash setup_wsl.sh
```
该脚本会自动：
1. 安装 `uv`（如果未安装）。
2. 创建虚拟环境 `.venv`。
3. 使用阿里云镜像安装 `vllm`、`transformers` 等依赖。
4. 从镜像站下载 `Qwen/Qwen3-0.6B` 模型到 `Task0/data`。

### 2. 启动 vLLM API 服务
执行以下脚本启动 OpenAI 兼容的 API 服务：
```bash
bash run_vllm_api.sh
```

### 3. 调用 API 测试
服务启动后，在另一个 WSL 终端或宿主机使用 `curl` 进行测试：
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "Qwen/Qwen3-0.6B",
		"messages": [
			{
				"role": "user",
				"content": "What is the capital of France?"
			}
		]
	}'
```

### 4. 运行 Transformers 对比脚本
在 WSL 终端执行：
```bash
source .venv/bin/activate
python3 transformers_inference.py --model ../data/Qwen3-0.6B
```

## 效率对比分析

| 特性 | Transformers (Native) | vLLM |
| :--- | :--- | :--- |
| **吞吐量 (Throughput)** | 较低 | 极高 (通常提升 10-20 倍) |
| **显存管理** | 静态分配，碎片化严重 | PagedAttention，动态分配 |
| **Batching 策略** | Static Batching | Continuous Batching |

### 为什么 vLLM 更快？

1. **PagedAttention**:
   - **问题**: 在 Transformers 中，KV Cache 需要连续的显存空间，导致预分配空间过大（预留最大长度）或产生大量碎片。
   - **解决**: vLLM 将 KV Cache 分块存储（类似操作系统的分页），允许在非连续显存中存储，几乎消除了显存碎片，使得系统可以容纳更大的 Batch Size。

2. **Continuous Batching (迭代级调度)**:
   - **问题**: 静态 Batching 必须等待 Batch 中最长的序列生成结束，短序列会浪费计算资源（Padding）。
   - **解决**: vLLM 在每个迭代步都会检查是否有新请求加入或旧请求结束，实现“随到随走”，极大提升了 GPU 利用率。

## 常见问题排查 (Troubleshooting)

1. **WSL 显存不足**:
   - 脚本已针对 12GB 显存优化（`--gpu-memory-utilization 0.3`）。如果仍报错，请关闭浏览器硬件加速或其他占用显存的程序。
   - 检查 WSL 是否能识别 GPU：执行 `nvidia-smi`。

2. **网络连接问题**:
   - 如果下载模型缓慢，请确保 `HF_ENDPOINT=https://hf-mirror.com` 已生效。
