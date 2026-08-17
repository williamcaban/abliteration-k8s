# NVIDIA CUDA 13.3.1 + cuDNN 9 on Red Hat UBI9
# Confirmed present in base image:
#   /usr/local/cuda/lib64/libcudart.so.13   (CUDA 13.3.1 runtime)
#   /usr/lib64/libcudnn.so.9               (cuDNN 9.24.0)
#   python3.11 available via ubi-9-appstream-rpms
FROM nvidia/cuda:13.3.1-cudnn-runtime-ubi9

# PyTorch CUDA wheel variant — cu126 wheels run on CUDA 13.x via backward
# compatibility (CUDA driver forwards API calls to newer runtime).
# Override at build time: --build-arg TORCH_CUDA_INDEX=cu130
ARG TORCH_CUDA_INDEX=cu126

# Install Python 3.11 and git; clean dnf cache to keep layer small
RUN dnf install -y python3.11 python3.11-pip python3.11-devel git gcc \
    && dnf clean all \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3.11    /usr/local/bin/pip3 \
    && ln -sf /usr/bin/pip3.11    /usr/local/bin/pip

# Scripts live at /opt/abliterator — outside the /workspace PVC mount so
# they are not shadowed when the PVC is attached at runtime.
WORKDIR /opt/abliterator

# Upgrade pip — the python3.11 shipped in UBI9 bundles an old pip that
# cannot satisfy flit_core>=3.11 required by modern PyTorch wheels.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch first (large wheel; separate layer for better cache reuse)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}

# Remaining deps — matches NousResearch requirements.txt + safetensors
RUN pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    datasets \
    huggingface-hub \
    matplotlib \
    pandas \
    pyyaml \
    safetensors \
    tqdm \
    transformers

# Clone tool at build time for reproducibility
RUN git clone --depth 1 \
    https://github.com/NousResearch/llm-abliteration.git \
    /opt/abliterator/llm-abliteration

# Copy helper scripts
COPY auto_yaml.py   /opt/abliterator/auto_yaml.py
COPY entrypoint.sh  /opt/abliterator/entrypoint.sh
RUN chmod +x /opt/abliterator/entrypoint.sh

# /workspace is the PVC mount point — create it so it exists at startup
RUN mkdir -p /workspace

# OpenShift compatibility: GID 0 on both dirs so any assigned UID can write
RUN chown -R 1001:0 /opt/abliterator /workspace && \
    chmod -R g=u    /opt/abliterator /workspace

USER 1001

# HuggingFace cache lands on the workspace PVC when /workspace is mounted
ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    TRITON_CACHE_DIR=/workspace/triton_cache \
    HOME=/workspace \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/opt/abliterator/entrypoint.sh"]
