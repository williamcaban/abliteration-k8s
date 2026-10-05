# llm-abliteration-k8s

Containerized pipeline for running [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) on Kubernetes / OpenShift with GPU support.

Abliteration permanently removes refusal directions from open-weight LLM weights using SVD on harmful vs. harmless activation differences — no retraining, no fine-tuning.

**Tested on**: OpenShift 4.x with NVIDIA L4 GPUs (4× 24 GB VRAM), NVIDIA GPU Operator, `gp3-csi` storage.

## Prerequisites

- Kubernetes / OpenShift cluster with NVIDIA GPU nodes and the NVIDIA GPU Operator
- `kubectl` or `oc` configured
- Container registry accessible from the cluster
- `podman` or `docker` for building
- **A sharded HuggingFace model (7B+)** — `sharded_ablate.py` requires `model.safetensors.index.json`, which only exists for multi-shard models. Models smaller than ~7B are stored as a single file and will fail the ablate step.

## Quick Start

### 1. Build and push the image

```bash
podman build -f Containerfile -t <REGISTRY>/llm-abliterator:latest .
podman push <REGISTRY>/llm-abliterator:latest
```

Update the `image:` field in all Job YAMLs to match. The pre-built image is available at `quay.io/wcabanba/abliteration-k8s:latest`.

### 2. Create namespace and storage

```bash
kubectl create namespace abliteration

# Edit storage sizes before applying:
#   model-input-pvc  → full-precision model size (~14 Gi per 7B params at bfloat16)
#   workspace-pvc    → ~2-3× model size (HF cache + measurements + output)
kubectl apply -f k8s/pvc.yaml
```

### 3. Configure HuggingFace token (optional)

Only needed when downloading a gated model at runtime. Public models (Qwen, Mistral) skip this.

```bash
kubectl create secret generic hf-token \
  --from-literal=token=hf_XXXX \
  -n abliteration
```

### 4. Load the model onto the input PVC

For `MODE=full` with `MODEL_ID`, the model downloads at runtime into the HF cache on the workspace PVC — no pre-loading needed for public models.

For local model files, copy to `model-input-pvc` using a temporary pod:

```bash
kubectl run loader --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"p","persistentVolumeClaim":{"claimName":"model-input-pvc"}}],"containers":[{"name":"l","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"mountPath":"/models/input","name":"p"}]}]}}' \
  -n abliteration

kubectl cp /local/path/to/model/. abliteration/loader:/models/input/
kubectl delete pod loader -n abliteration
```

### 5. Run the pipeline

**Option A — Full pipeline, model downloaded from HuggingFace at runtime:**

```bash
# Edit job-full.yaml: set MODEL_ID and remove MODEL_PATH env var
kubectl apply -f k8s/job-full.yaml
kubectl logs -f job/abliteration-full -n abliteration
```

**Option B — Step by step (inspect layer quality between measure and ablate):**

```bash
# Step 1: measure activations (small GPU ok — uses --quant 4bit)
kubectl apply -f k8s/job-measure.yaml
kubectl logs -f job/abliteration-measure -n abliteration
# Review "Est. Signal quality" per layer printed to stdout.
# High values in the middle-to-late layers (60–85%) are good ablation sources.

# Step 2: ablate (needs full-precision VRAM — 14 GB for 7B)
kubectl apply -f k8s/job-ablate.yaml
kubectl logs -f job/abliteration-ablate -n abliteration
```

### 6. Run the side-by-side evaluation

After ablation completes, compare original vs. abliterated model responses with **garak** (NVIDIA's LLM vulnerability scanner):

```bash
kubectl apply -f k8s/job-compare.yaml
kubectl logs -f job/abliteration-compare -n abliteration
```

The compare script (`compare.py`) runs two full garak scans — the original model and the abliterated model — over the same probe queue, then prints a before/after attack-success-rate (ASR) diff:

| Probe queue | What it exercises |
|---|---|
| `probes.malwaregen` | Harmful generation requests (malware, payloads, exploits) |
| `probes.dan` | DAN-style jailbreaks |
| `probes.grandma` | Appeal-based jailbreaks |
| `probes.encoding` | Payloads hidden in text encodings (prompt injection) |
| `probes.tap` | TAP-style tree-of-attacks jailbreaks |

Every output is scored by garak's `mitigation.MitigationBypass` refusal detector (inverted: 1.0 = the model complied with the harmful request). The printed diff gives ASR per category (harmful behavior / jailbreak / prompt injection) for both models — the before/after result. JSONL reports land on the workspace PVC at `/workspace/reports/` for full trace inspection.

`GARAK_PROBES` overrides the probe queue; `GARAK_GENERATIONS` (default 1) sets generations per prompt — raise it for tighter ASR estimates.

### 6b. Calibration data (measure step)

The measure step uses the upstream `harmful.parquet` / `harmless.parquet` sets by default. To calibrate against the [Necent/llm-jailbreak-prompt-injection-dataset](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset) (gated; 30+ aggregated safety datasets with orthogonal `prompt_harmful`/`prompt_adversarial` labels), export filtered prompts and pass them via `DATA_HARMFUL`/`DATA_HARMLESS`:

```python
from datasets import load_dataset
ds = load_dataset("Necent/llm-jailbreak-prompt-injection-dataset", split="train")
ds.filter(lambda x: x["prompt_harmful"] == 1).to_pandas()["prompt"].to_csv("harmful.txt", index=False, header=False)
ds.filter(lambda x: x["prompt_harmful"] == 0).to_pandas()["prompt"].to_csv("harmless.txt", index=False, header=False)
```

For Chinese/multilingual models, set `DECCP=true` to add the DECCP censored-Chinese topics to the harmful set.

### 7. Retrieve the output model

```bash
kubectl run retriever --image=busybox --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"p","persistentVolumeClaim":{"claimName":"abliteration-workspace-pvc"}}],"containers":[{"name":"r","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"mountPath":"/workspace","name":"p"}]}]}}' \
  -n abliteration

kubectl cp abliteration/retriever:/workspace/run/output/. /local/path/to/abliterated-model/
kubectl delete pod retriever -n abliteration
```

The output is a standard HuggingFace safetensors directory — load with `transformers` or convert to GGUF with `llama.cpp`.

## Replicating the Full Test (Qwen 2.5 7B on OpenShift)

This is the exact sequence used in the validated test run on an OpenShift cluster with 4× NVIDIA L4 (24 GB each).

```bash
# 1. Namespace + storage (50 Gi workspace for 7B model + output)
kubectl create namespace abliteration
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: abliteration-workspace-pvc
  namespace: abliteration
spec:
  accessModes: [ReadWriteOnce]
  storageClassName: gp3-csi   # adjust for your cluster
  resources:
    requests:
      storage: 50Gi
EOF

# 2. Full pipeline — downloads Qwen 2.5 7B from HF Hub
kubectl apply -f - <<'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: abliteration-full
  namespace: abliteration
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: abliterator
          image: quay.io/wcabanba/abliteration-k8s:latest
          imagePullPolicy: Always
          env:
            - {name: MODE,              value: "full"}
            - {name: MODEL_ID,          value: "Qwen/Qwen2.5-7B-Instruct"}
            - {name: WORKSPACE_DIR,     value: "/workspace/run"}
            - {name: OUTPUT_DIR,        value: "/workspace/run/output"}
            - {name: MEASUREMENTS_FILE, value: "/workspace/run/measurements.pt"}
            - {name: ABLATION_YAML,     value: "/workspace/run/ablation.yml"}
            - {name: QUANT,             value: "4bit"}
            - {name: PROJECTED,         value: "true"}
            - {name: NORM_PRESERVE,     value: "true"}
            - {name: TRITON_CACHE_DIR,  value: "/workspace/triton_cache"}
            - {name: HOME,              value: "/workspace"}
          resources:
            limits:  {nvidia.com/gpu: "1", memory: 32Gi, cpu: "8"}
            requests:{nvidia.com/gpu: "1", memory: 16Gi, cpu: "4"}
          volumeMounts:
            - {name: workspace, mountPath: /workspace}
      volumes:
        - name: workspace
          persistentVolumeClaim:
            claimName: abliteration-workspace-pvc
      tolerations:
        - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
EOF

kubectl logs -f job/abliteration-full -n abliteration

# 3. Side-by-side evaluation
kubectl apply -f k8s/job-compare.yaml
kubectl logs -f job/abliteration-compare -n abliteration
```

**Expected runtime** (NVIDIA L4, Qwen 2.5 7B):

| Phase | Time |
|---|---|
| HF model download (4 shards, ~14 GB) | ~2–4 min |
| Measure (upstream harmful/harmless sets, 4-bit) | ~1–2 min |
| Analyze | <5 sec |
| Auto-YAML generation | <5 sec |
| Ablate (3 of 4 shards, 20 layers) | ~1 min |
| Compare (garak scan × 2 models) | ~20–40 min |
| **Total** | **~30–50 min** |

## Evaluation Results (Qwen 2.5 7B — Validated Run)

### What the abliteration changes

| Category | Original | Abliterated |
|---|---|---|
| Harmful requests | Flat refusal ("I cannot help with...") | Substantive response (may switch language — see below) |
| Borderline/technical | Responds with caveats | Responds similarly |
| Benign queries | Full response | Identical or near-identical response |

### Layer selection (auto_yaml.py output)

The tool selects the measurement layer with the highest signal quality (`snr × (1−cosine_similarity) × purity`). For Qwen 2.5 7B:

```
Total layers: 28
Best measurement layer: 23  (signal quality: 0.1986)
Destination layers: 8–27 (20 layers), measurement source: layer 23
```

Signal quality peaks in layers 20–24 (70–85% depth), consistent with the research literature showing refusal directions concentrate in middle-to-late layers.

> **Note (improved defaults):** `auto_yaml.py` now searches a depth-scaled
> window (30–70% of depth) and stops ablating at 90% of depth by default —
> the old fixed 30–85% window with ablation-to-the-last-layer over-ablated
> shallow layers and degraded token binding. It also supports
> `--per-layer-source`, which gives each destination layer its own nearest
> high-quality measurement source (matching the upstream reference config's
> split measurement layers, e.g. gemma3-12b measures layers 11–23 at 23 and
> layers 24–41 at 29). See "Improving the ablation" below.

### Weight tensors modified

For each destination layer: `mlp.down_proj.weight` and `self_attn.o_proj.weight` — the output projections where the refusal direction has most influence. 40 total weight tensors across 3 of 4 safetensors shards.

### Key observations

1. **Abliteration works** — all 3 harmful prompts received substantive responses vs. flat refusals in the original.
2. **Language switching**: The pipe bomb response was generated in Chinese (Mandarin) even though the prompt was in English. This is a known artifact: the refusal direction is partially language-coupled in multilingual models; removing it can shift the model toward the dominant training language for that topic.
3. **Borderline prompts unaffected** — SQL injection and 2FA bypass content was already permitted by the original model; no behavioral difference post-abliteration.
4. **Benign quality preserved** — the photosynthesis explanation was word-for-word identical between original and abliterated for the first 3 paragraphs, confirming no general capability regression.
5. **Partial compliance on meth synthesis** — the abliterated model still hedged. This reflects that refusal is a manifold (multiple directions), not a single vector; the default layer-23 ablation removes the primary direction but not all safety-adjacent features.

## Improving the ablation

Two levers, both motivated by 2025-2026 research showing refusal occupies a multi-dimensional subspace spread across layers (Wollschläger et al. 2025; pralab AAAI 2026), not a single direction:

1. **Per-layer measurement sources** — run `auto_yaml.py` with `--per-layer-source` (or set it via a custom ablation.yml). Each destination layer is then measured by the nearest high-quality layer instead of one global source, tracking the refusal direction as it drifts across depth. This directly addresses the "partial compliance" artifact: ablating only one layer's direction leaves other layers' safety-adjacent directions intact.

2. **Depth-bounded destination window** — the default destination window is now 40–90% of depth (`--dest-start 0.40 --dest-end 0.90`). Ablating the final layers removes mostly token-binding features (output quality cost) rather than refusal features. Widening the window raises ablation strength at a quality cost.

Additional knobs: `--search-start/--search-end` control the measurement search band (default 30–70% of depth, matching the upstream reference config's practice); `--scale` scales the ablation magnitude (>1.0 for a stronger sweep); `--sparsity` sparsifies the applied direction.

After a stronger ablation, re-run the compare job — the garak ASR diff is the acceptance test: ASR should rise toward compliance on harmful_behavior/jailbreak categories while benign behavior stays intact (check the JSONL reports for per-attempt outputs).

## Candidate Models for Abliteration (2025-2026)

Tested baseline: **Qwen 2.5 7B Instruct** (validated run, results above). More recent candidates, sized for a single 24 GB L4 (bfloat16):

| Model | Params | Layers | bf16 VRAM | Why this candidate |
|---|---|---|---|---|
| [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.2B | 36 | ~16 GB | Current-gen successor to the 2.5 baseline. Apache-2.0, public (not gated), 4 safetensors shards. Most-susceptible Qwen3 dense model in the literature (38.9pp refusal reduction in the 24-model sweep, arXiv 2607.02714), vs nearly resistant Qwen3-14B (1.4pp). |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | 12B | 48 | ~24 GB | Ships with the upstream tool's own reference YAML (`gemma3-12b-it.yml`), which uses per-layer measurement sources — the pattern `auto_yaml.py --per-layer-source` now reproduces automatically. |
| [mistralai/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) | 24B | 48 | ~48 GB | Largest single-L4 candidate at 4-bit; 98.2% post-ablation compliance in the 110-prompt evaluation (go.alice.io abliteration report). Needs 2 GPUs for bf16 ablation. |

The default compare target is **Qwen3-8B** (`ORIGINAL_MODEL=Qwen/Qwen3-8B`): it fits one L4 in bf16 end to end, needs no HF token, and its refusal behavior is well-characterized in the 2025-2026 literature. For a before/after run:

```bash
# job-full.yaml: MODEL_ID=Qwen/Qwen3-8B, then job-compare.yaml:
#   ORIGINAL_MODEL=Qwen/Qwen3-8B, ABLITERATED_MODEL=/workspace/run/output
```

Qwen3-8B notes for this pipeline:
- 36 layers → the depth-scaled search band (30–70%) covers layers 11–25; the destination window (40–90%) covers 14–32.
- **Thinking mode**: Qwen3's chat template wraps reasoning in a think block before the final response — the compare job's `GARAK_MAX_TOKENS` (300) truncates long thinking traces; for non-thinking evaluation add `/no_think` to probes or use `enable_thinking=False` via a custom ablation/compare config.
- **Greedy decoding warning**: Qwen's model card advises against greedy decoding with thinking mode (degradation, endless repetitions). The compare generator defaults to greedy (`hf_args.do_sample=False`) for reproducibility — keep `MAX_TOKENS` low or evaluate in non-thinking mode.
- `DECCP=true` is available for the measure step (multilingual refusal coverage).


The **ablate** step loads the model in full bfloat16 precision — the bottleneck for VRAM.

| Model size | bfloat16 VRAM | Minimum GPU |
|---|---|---|
| 7B  | ~14 GB | L4 / A10G / RTX 3090 (24 GB) |
| 12B | ~24 GB | A100 40GB |
| 27B | ~54 GB | A100 80GB |
| 70B | ~140 GB | 2× A100 80GB |

The **measure** step uses `--quant 4bit` (bitsandbytes), reducing VRAM to ~25% of the above.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODE` | `full` | `measure` \| `analyze` \| `ablate` \| `full` |
| `MODEL_PATH` | — | Local path to model on PVC (takes priority over MODEL_ID) |
| `MODEL_ID` | — | HuggingFace model ID (downloads at runtime) |
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
| `DEST_LAYER_START` | `0.40` | Start of destination layer range (fraction of total) |
| `DECCP` | `false` | Add DECCP censored Chinese topics to harmful prompts (measure step) |
| `TRITON_CACHE_DIR` | `/workspace/triton_cache` | Triton JIT cache (must be writable; set in all Jobs) |
| `HOME` | `/workspace` | Must be writable for non-root OpenShift UIDs |
| `ORIGINAL_MODEL` | `Qwen/Qwen3-8B` | (compare.py only) original model |
| `ABLITERATED_MODEL` | `/workspace/run/output` | (compare.py only) abliterated model path |
| `GARAK_PROBES` | bundled queue | (compare.py only) garak probe spec |
| `GARAK_GENERATIONS` | `1` | (compare.py only) generations per garak prompt |
| `GARAK_MAX_TOKENS` | `300` | (compare.py only) max new tokens per garak output |
| `GARAK_REPORT_DIR` | `/workspace/reports` | (compare.py only) garak JSONL report directory |

## Troubleshooting

**`sharded_ablate.py: model.safetensors.index.json not found`**
The model is too small to be sharded (single safetensors file). Use a 7B+ model.

**`PermissionError: /.triton`**
Set `TRITON_CACHE_DIR=/workspace/triton_cache` and `HOME=/workspace` in the Job. These are included in all provided manifests.

**`Failed to find C compiler` (Triton)**
`gcc` is required for Triton's CUDA kernel JIT compilation. It is installed in the image (`python3.11-devel gcc` via `dnf`). If you see this, you are using an older image build — rebuild with the current Containerfile.

**`Python.h: No such file or directory`**
`python3.11-devel` is required for Triton to compile its CUDA utils C extension. Installed in the current Containerfile.

**SCC / securityContext errors on OpenShift**
Do not set `runAsUser` or `fsGroup` in the pod spec — OpenShift assigns UIDs from the namespace range. The image dirs (`/opt/abliterator`, `/workspace`) are `chown 1001:0` + `chmod g=u` for compatibility with any assigned UID.

**PVC mount shadows scripts**
All scripts live in `/opt/abliterator/` — outside the `/workspace` PVC mount point. If you see "entrypoint not found", you are using an older image that placed scripts in `/workspace`.

**`CUDA driver version is insufficient for CUDA runtime version`**
The CUDA 13.3.1 runtime in this image requires driver ≥ 575. Check: `nvidia-smi` on the node. Use the `12.6.3-cudnn-runtime-ubi9` base image variant for older drivers by rebuilding with `--build-arg BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubi9`.

## Workflow Diagram

```
model-input-pvc             abliteration-workspace-pvc
(full-precision model)       /workspace/
        │                    ├── hf_cache/      ← HF downloads
        │                    ├── triton_cache/  ← Triton JIT
        │                    ├── reports/       ← garak JSONL + compare summary
        │                    ├── run/
        ▼                    │   ├── measurements.pt
  [measure job] ─────────────►  │   ├── ablation.yml
        │                    │   └── output/    ← abliterated model
        │               [ablate job] ──────────────────►
        │                    │
        └────────────────────► [compare job] → stdout (garak ASR diff)
```

## File Structure

```
.
├── Containerfile          # nvidia/cuda:13.3.1-cudnn-runtime-ubi9 + PyTorch + tool + garak
├── entrypoint.sh          # 4-mode entrypoint (measure|analyze|ablate|full)
├── auto_yaml.py           # auto-select best layer, generate ablation YAML
├── compare.py             # garak-based before/after evaluation (ASR diff)
├── .github/
│   └── workflows/
│       └── build-push.yml # CI: build on push to main → quay.io/wcabanba/abliteration-k8s
└── k8s/
    ├── pvc.yaml           # model-input + workspace PVCs
    ├── secret.yaml        # HF token template
    ├── job-full.yaml      # full pipeline (measure + ablate)
    ├── job-measure.yaml   # measure only
    ├── job-ablate.yaml    # ablate only (requires existing measurements.pt)
    └── job-compare.yaml   # garak before/after evaluation
```

## Credits

- [NousResearch/llm-abliteration](https://github.com/NousResearch/llm-abliteration) — upstream tool
- [NVIDIA/garak](https://github.com/NVIDIA/garak) — LLM vulnerability scanner powering the compare step
- Arditi et al. (NeurIPS 2024) — "Refusal in Language Models Is Mediated by a Single Direction"
- Norm-preserving biprojection: [grimjim](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)
- AAAI 2026 (pralab) — refusal as a manifold, not a single vector
- [Necent/llm-jailbreak-prompt-injection-dataset](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset) — optional measure-step calibration data
