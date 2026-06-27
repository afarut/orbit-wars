# CUDA dev-образ для Orbit Wars SFT.
# База несёт CUDA/cuDNN рантайм; torch ставится из PyPI (на linux это CUDA-сборка).
# Запускать ТОЛЬКО на хосте с NVIDIA-драйвером + nvidia-container-toolkit (--gpus all).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.12 (нативный в 24.04) + сборочные инструменты для редких sdist-зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Изолированный venv с CUDA-torch; кладём его в PATH, чтобы `python`/`pip` били в него,
# а не в хостовый .venv (CPU-сборка), который примонтируется поверх кода в /workspace.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Зависимости — отдельным слоем, чтобы кэш переживал правки кода.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

WORKDIR /workspace
# Код НЕ COPY-им — он монтируется volume'ом из compose (живые правки без пересборки).
CMD ["sleep", "infinity"]
