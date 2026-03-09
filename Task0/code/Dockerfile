FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_CACHE_DIR=/app/.uv_cache
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 切换 Ubuntu 源为中科大源 (USTC)
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# 安装基础依赖
# 使用 --fix-missing 忽略临时性的 404 错误
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p /app/data /app/code /app/.uv_cache

# 将本地 bin 加入 PATH
ENV PATH="/usr/local/bin:${PATH}"

# 默认命令
CMD ["/bin/bash"]
