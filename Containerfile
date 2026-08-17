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
RUN dnf install -y python3.11 python3.11-pip git \
    && dnf clean all \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3.11    /usr/local/bin/pip3 \
    && ln -sf /usr/bin/pip3.11    /usr/local/bin/pip

WORKDIR /workspace

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
    /workspace/llm-abliteration

# Copy helper scripts
COPY auto_yaml.py   /workspace/auto_yaml.py
COPY entrypoint.sh  /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# OpenShift compatibility: UID 1001, GID 0 so any assigned UID can write
RUN chown -R 1001:0 /workspace && chmod -R g=u /workspace

USER 1001

# HuggingFace cache lands on the workspace PVC when /workspace is mounted
ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/workspace/entrypoint.sh"]
