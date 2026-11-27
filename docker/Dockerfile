# syntax=docker/dockerfile:1
# Build only the committed fork. Datasets, local settings and credentials in the
# host checkout are never part of the source copied into an image.
FROM scratch AS source
ARG GIT_COMMIT=main
ADD https://github.com/BlipOnNobodysRadar/ai-toolkit.git#${GIT_COMMIT} /src/

# runtime (not devel) is enough: torch/flash-attn/natten are all prebuilt
# wheels that bundle their CUDA libs, and triton JITs with its own ptxas.
# Host requirement: NVIDIA driver >= 580 (CUDA 13) to run the cu130 wheels.
FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04

LABEL authors="jaret"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV CHECKPOINT_DISABLE=1 NEXT_TELEMETRY_DISABLED=1

# ref https://en.wikipedia.org/wiki/CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    unzip \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_24.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set aliases for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch before cache bust to avoid redownloading pytorch
# (versions must match manager/spec.py — the AI Toolkit Manager's linux spec)
RUN pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

WORKDIR /app/ai-toolkit

# ---------------------------------------------------------------------------- #
# Dependency layers come BEFORE the application source so they are only rebuilt (and
# only need to be re-pulled by servers) when the dependency manifests change,
# not on every code change.
# ---------------------------------------------------------------------------- #

# Install Python dependencies (only re-runs when the requirements files change)
COPY --from=source /src/requirements.txt /src/requirements_base.txt /app/ai-toolkit/
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install setuptools==69.5.1 --no-cache-dir --break-system-packages

# Accelerators, matching the manager's linux cu130 spec (manager/spec.py):
# flash-attn 2.8.3 (prebuilt for torch 2.13 / cu130 / cp312), NATTEN 0.21.7,
# and torchcodec 0.15. Installed AFTER requirements with -U so they override
# any older pins in there (same order the manager uses).
RUN pip install --no-cache-dir --break-system-packages -U \
    torchcodec==0.15.0 \
    natten==0.21.7+torch2130cu130 --find-links https://whl.natten.org \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl && \
    python -c "import flash_attn, natten, torchcodec; print('accelerators OK:', flash_attn.__version__, natten.__version__, torchcodec.__version__)"

# flash-linear-attention (DeltaNet kernels for Qwen3.5/3.8), matching manager/spec.py FLA.
# Pure-Python Triton package installed bare and WITHOUT -U: its unpinned transformers
# requirement must not upgrade the pinned transformers from requirements_base.txt.
RUN pip install --no-cache-dir --break-system-packages flash-linear-attention==0.5.2 && \
    python -c "import fla, transformers; print('fla OK:', fla.__version__, 'transformers', transformers.__version__)"

# Install Node dependencies (only re-runs when package.json / package-lock.json change)
COPY --from=source /src/ui/package.json /src/ui/package-lock.json /app/ai-toolkit/ui/
RUN cd /app/ai-toolkit/ui && npm ci --no-audit

COPY --from=source /src/ /app/ai-toolkit/

# Build UI (re-runs on code change, but reuses the cached node_modules above).
# update_db runs first because it does `prisma generate`, which creates the
# @prisma/client types the TS build needs. In the old layout generate happened
# as a side effect of npm install seeing the schema; now the source arrives
# after npm ci, so run it explicitly before the build.
RUN cd /app/ai-toolkit/ui && \
    npm run update_db && \
    npm run build

# Expose port (assuming the application runs on port 3000)
EXPOSE 8675

WORKDIR /

COPY --from=source /src/docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
