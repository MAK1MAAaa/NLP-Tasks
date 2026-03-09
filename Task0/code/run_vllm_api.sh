#!/bin/bash
source .venv/bin/activate

MODEL_DIR="../data/Qwen3-0.6B"
MODEL_NAME="Qwen/Qwen3-0.6B"

echo "Starting vLLM API Server on port 8000..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization 0.3 \
    --max-model-len 2048 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code
