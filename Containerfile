FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Install git for repo clone; clean apt lists to keep image small
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps — matches requirements.txt plus safetensors (used by sharded_ablate.py)
RUN pip install --no-cache-dir \
    accelerate \
    bitsandbytes \
    datasets \
    huggingface-hub \
    matplotlib \
    pandas \
    pyyaml \
    safetensors \
    torch \
    tqdm \
    transformers

# Clone tool at build time for reproducibility
RUN git clone --depth 1 https://github.com/NousResearch/llm-abliteration.git /workspace/llm-abliteration

# Copy helper scripts
COPY auto_yaml.py /workspace/auto_yaml.py
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# OpenShift compatibility: UID 1001, GID 0 so arbitrary UID can write
RUN chown -R 1001:0 /workspace && chmod -R g=u /workspace

USER 1001

# HuggingFace cache lands on the workspace PVC when /workspace is mounted
ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/workspace/entrypoint.sh"]
