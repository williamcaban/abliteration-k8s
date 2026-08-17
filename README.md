# llm-abliteration-k8s

Containerized pipeline for running [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) on Kubernetes / OpenShift with GPU support.

Abliteration permanently removes refusal directions from open-weight LLM weights using SVD on harmful vs. harmless activation differences — no retraining, no fine-tuning.

## Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes (or OpenShift/RHOAI)
- `kubectl` / `oc` configured
- Container registry accessible from the cluster
- `podman` or `docker` for building the image
- A full-precision HuggingFace model directory (safetensors format)

## Quick Start

### 1. Build and push the image

```bash
podman build -f Containerfile -t <REGISTRY>/llm-abliterator:latest .
podman push <REGISTRY>/llm-abliterator:latest
```

Replace `<REGISTRY>` with your registry (e.g. `quay.io/yourorg`). Update the `image:` field in all Job YAMLs to match.

### 2. Create namespace and storage

```bash
kubectl create namespace abliteration

# Edit k8s/pvc.yaml storage sizes to match your model before applying:
#   model-input-pvc  → full-precision model size (14 Gi per 7B params at bf16)
#   workspace-pvc    → ~2x model size (measurements + output model)
kubectl apply -f k8s/pvc.yaml
```

### 3. Configure HuggingFace token (optional)

Only needed if downloading the model at runtime via `MODEL_ID`. Skip if pre-loading the model onto the PVC.

```bash
kubectl create secret generic hf-token \
  --from-literal=token=hf_XXXX \
  -n abliteration
```

### 4. Load the model onto the input PVC

Copy a full-precision HuggingFace model directory (safetensors files + config.json + tokenizer files) onto `model-input-pvc`. One approach using a temporary pod:

```bash
kubectl run model-loader --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"pvc","persistentVolumeClaim":{"claimName":"model-input-pvc"}}],"containers":[{"name":"model-loader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"mountPath":"/models/input","name":"pvc"}]}]}}' \
  -n abliteration

# From another terminal:
kubectl cp /local/path/to/model/. abliteration/model-loader:/models/input/
kubectl delete pod model-loader -n abliteration
```

### 5. Run the pipeline

**Option A — Full pipeline in one shot** (recommended for first run):

```bash
kubectl apply -f k8s/job-full.yaml
kubectl logs -f job/abliteration-full -n abliteration
```

**Option B — Step by step** (recommended when tuning ablation parameters):

```bash
# Step 1: measure activations (runs with 4-bit quant, smaller GPU)
kubectl apply -f k8s/job-measure.yaml
kubectl logs -f job/abliteration-measure -n abliteration
# Review the per-layer signal quality printed to stdout.
# Look for layers with high Est. Signal Quality in the middle-to-late range.

# Step 2: ablate (uses full-precision model; size GPU accordingly)
kubectl apply -f k8s/job-ablate.yaml
kubectl logs -f job/abliteration-ablate -n abliteration
```

### 6. Retrieve the output model

The abliterated model is written to `workspace-pvc` at `/workspace/run/output/`. Copy it out the same way you loaded the input:

```bash
kubectl run model-retriever --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"pvc","persistentVolumeClaim":{"claimName":"abliteration-workspace-pvc"}}],"containers":[{"name":"r","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"mountPath":"/workspace","name":"pvc"}]}]}}' \
  -n abliteration

kubectl cp abliteration/model-retriever:/workspace/run/output/. /local/path/to/abliterated-model/
kubectl delete pod model-retriever -n abliteration
```

The output is a standard HuggingFace safetensors directory — load it with `transformers` or convert to GGUF with `llama.cpp`.

## GPU Sizing

The **ablate** step loads the model in full bf16 precision and is the bottleneck.

| Model size | bf16 VRAM | Minimum GPU |
|---|---|---|
| 7B | ~14 GB | RTX 4080 / A10 (16 GB) |
| 12B | ~24 GB | A10G / RTX 3090 (24 GB) |
| 27B | ~54 GB | A100 80GB |
| 70B | ~140 GB | 2× A100 80GB |

The **measure** step runs with `--quant 4bit` (bitsandbytes), reducing VRAM to ~25% of the above. A smaller GPU node pool can be used for measure-only jobs.

## Environment Variables

All parameters are set via env vars in the Job manifests.

| Variable | Default | Description |
|---|---|---|
| `MODE` | `full` | `measure` \| `analyze` \| `ablate` \| `full` |
| `MODEL_PATH` | — | Local path to model on PVC (takes priority over MODEL_ID) |
| `MODEL_ID` | — | HuggingFace model ID (downloads at runtime; needs HF_TOKEN) |
| `WORKSPACE_DIR` | `/workspace/run` | Dir for measurements, YAML, and output |
| `OUTPUT_DIR` | `$WORKSPACE_DIR/output` | Where the abliterated model is saved |
| `MEASUREMENTS_FILE` | `$WORKSPACE_DIR/measurements.pt` | Path to measurements file |
| `ABLATION_YAML` | `$WORKSPACE_DIR/ablation.yml` | Ablation config; auto-generated if absent |
| `QUANT` | _(none)_ | `4bit` or `8bit` for the measure step |
| `DATA_HARMFUL` | _(built-in)_ | Custom harmful prompts file |
| `DATA_HARMLESS` | _(built-in)_ | Custom harmless prompts file |
| `PROJECTED` | `true` | Orthogonalize refusal vs. harmless direction |
| `NORM_PRESERVE` | `true` | Preserve weight norms during ablation |
| `SCALE` | `1.0` | Ablation scale factor for auto-generated YAML |
| `SPARSITY` | `0.0` | Sparsity fraction for auto-generated YAML |
| `DEST_LAYER_START` | `0.30` | Start of destination layer range (fraction of total) |

## Custom Ablation YAML

For fine-grained control, mount a hand-crafted `ablation.yml` onto the workspace PVC at `$ABLATION_YAML` before running the ablate job. If the file exists, auto-generation is skipped.

YAML format (see `gemma3-12b-it.yml` in the upstream repo for a full example):

```yaml
model: /models/input           # or HF model ID
measurements: /workspace/run/measurements.pt
output: /workspace/run/output
ablate:
  - layer: 14
    measurement: 23            # source measurement layer
    scale: 1.0
    sparsity: 0.00
  - layer: 15
    measurement: 23
    scale: 1.0
    sparsity: 0.00
  # ... repeat for each destination layer
```

Run `analyze.py` output (available in the measure job logs) to identify layers with high **Est. Signal Quality**. Target middle-to-late layers (roughly 30–100% of model depth).

## Workflow Diagram

```
model-input-pvc          abliteration-workspace-pvc
(full-precision model)
        │
        ▼
  [measure job] ──────► measurements.pt
        │                       │
        │               [auto_yaml.py or
        │                hand-crafted YAML]
        │                       │
        │                       ▼
        └──────────────► [ablate job] ──► output/ (abliterated model)
```

## Important Notes

- **Cannot abliterate pre-quantized models.** GGUF, GPTQ, and AWQ models are not supported. Abliterate the full-precision model first, then quantize the output.
- **Abliteration amplifies quantization artifacts.** Test the abliterated model before quantizing.
- **MoE models** (Mixtral, DeepSeek-MoE, Llama 4) are supported. Use `--quant 4bit` for the measure step to fit large MoE models in VRAM.
- **Output format** is HuggingFace safetensors — compatible with vLLM, Ollama, and llama.cpp (after GGUF conversion).

## Credits

- [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) — upstream tool
- Arditi et al. (NeurIPS 2024) — "Refusal in Language Models Is Mediated by a Single Direction"
- Norm-preserving biprojection: [grimjim's blog](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
