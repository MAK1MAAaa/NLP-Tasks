#!/bin/bash
set -e

# 1. 构建镜像
echo "Building Docker image..."
docker build -t task0_vllm_uv -f Dockerfile .

# 2. 获取宿主机绝对路径 (兼容 WSL/Git Bash)
# 使用 pwd -W 获取 Windows 风格路径（如果是 Git Bash），或者直接用 $(pwd)
HOST_DATA_DIR="$(cd ../data && pwd)"
HOST_CODE_DIR="$(cd . && pwd)"
HOST_CACHE_DIR="$(cd .uv_cache && pwd || mkdir -p .uv_cache && cd .uv_cache && pwd)"

echo "Host Data Dir: $HOST_DATA_DIR"

# 3. 运行容器
echo "Starting Docker container..."
docker run --gpus all -it --rm \
    -p 8000:8000 \
    -v "$HOST_DATA_DIR:/app/data" \
    -v "$HOST_CODE_DIR:/app/code" \
    -v "$HOST_CACHE_DIR:/app/.uv_cache" \
    --shm-size 16g \
    task0_vllm_uv \
    /bin/bash -c "
        # 强制使用阿里云镜像，避免版本混乱
        export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/;
        export HF_ENDPOINT=https://hf-mirror.com;
        export UV_LINK_MODE=copy;

        echo '--- Disk Space Check ---'
        df -h /app/data

        echo '--- Installing Dependencies ---'
        uv pip install --system \
            'numpy<2' \
            'vllm==0.6.3' \
            'transformers>=4.45.0' \
            'huggingface-hub[cli]>=0.25.0'

        MODEL_NAME='Qwen/Qwen3-0.6B'
        MODEL_DIR='/app/data/Qwen3-0.6B'

        echo '--- Downloading Model ---'
        # 如果 config.json 不存在或大小为 0，则重新下载
        if [ ! -s \"\$MODEL_DIR/config.json\" ]; then
            echo \"Downloading \$MODEL_NAME to \$MODEL_DIR...\"
            # 使用 python 模块方式调用 cli 更加稳健
            python3 -m huggingface_hub.commands.cli download \"\$MODEL_NAME\" --local-dir \"\$MODEL_DIR\"
        fi

        echo '--- Starting vLLM API Server ---'
        if [ -s \"\$MODEL_DIR/config.json\" ]; then
            python3 -m vllm.entrypoints.openai.api_server \
                --model \"\$MODEL_DIR\" \
                --served-model-name \"\$MODEL_NAME\" \
                --gpu-memory-utilization 0.3 \
                --max-model-len 2048 \
                --host 0.0.0.0 \
                --port 8000 \
                --trust-remote-code
        else
            echo 'FATAL: Model download failed or disk is full.'
            df -h /app/data
            exit 1
        fi
    "
