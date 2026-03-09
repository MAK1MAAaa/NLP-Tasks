#!/bin/bash
set -e

# 1. 安装 uv (如果未安装)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 2. 创建虚拟环境
echo "Creating virtual environment..."
uv venv .venv
source .venv/bin/activate

# 3. 设置镜像源并安装依赖
echo "Installing dependencies..."
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
uv pip install \
    'numpy<2' \
    'vllm>=0.6.0' \
    'transformers>=4.45.0' \
    'torch>=2.4.0' \
    'huggingface-hub[cli]>=0.25.0'

# 4. 下载模型
MODEL_NAME="Qwen/Qwen3-0.6B"
MODEL_DIR="../data/Qwen3-0.6B"

mkdir -p ../data

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading $MODEL_NAME..."
    export HF_ENDPOINT=https://hf-mirror.com
    uv run huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR"
fi

echo "Setup complete! You can now run the inference scripts."
