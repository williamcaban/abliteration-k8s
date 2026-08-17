#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# NousResearch/llm-abliteration — K8s entrypoint
#
# Modes (set via MODE env var):
#   measure  — run measure.py only; saves .pt file to WORKSPACE_DIR
#   analyze  — run analyze.py on existing measurements (no chart; stdout only)
#   ablate   — run sharded_ablate.py; uses ABLATION_YAML or auto-generates one
#   full     — measure → analyze → auto-generate YAML → ablate (default)
#
# Required env vars (one of):
#   MODEL_PATH   — local path to full-precision HF model directory (preferred for K8s)
#   MODEL_ID     — HuggingFace model ID (downloaded at runtime; needs HF_TOKEN)
#
# Optional env vars:
#   MODE             — measure | analyze | ablate | full  (default: full)
#   WORKSPACE_DIR    — writable dir for measurements, YAML, output  (default: /workspace/run)
#   OUTPUT_DIR       — where abliterated model is saved (default: WORKSPACE_DIR/output)
#   MEASUREMENTS_FILE— path to .pt file (default: WORKSPACE_DIR/measurements.pt)
#   ABLATION_YAML    — path to ablation YAML; if absent in ablate/full mode, auto-generated
#   QUANT            — 4bit | 8bit for measure step (reduces VRAM; ablate always uses fp)
#   DATA_HARMFUL     — custom harmful prompts file (.txt/.json/.jsonl/.parquet)
#   DATA_HARMLESS    — custom harmless prompts file
#   PROJECTED        — true|false — orthogonalize refusal vs harmless direction (default: true)
#   NORM_PRESERVE    — true|false — preserve weight norms during ablation (default: true)
#   SCALE            — ablation scale factor for auto-generated YAML (default: 1.0)
#   SPARSITY         — sparsity for auto-generated YAML (default: 0.0)
#   DEST_LAYER_START — fraction of layers to start ablating from (default: 0.30)
# ---------------------------------------------------------------------------

MODE="${MODE:-full}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_ID="${MODEL_ID:-}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace/run}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/output}"
MEASUREMENTS_FILE="${MEASUREMENTS_FILE:-${WORKSPACE_DIR}/measurements.pt}"
ABLATION_YAML="${ABLATION_YAML:-${WORKSPACE_DIR}/ablation.yml}"
QUANT="${QUANT:-}"
DATA_HARMFUL="${DATA_HARMFUL:-}"
DATA_HARMLESS="${DATA_HARMLESS:-}"
PROJECTED="${PROJECTED:-true}"
NORM_PRESERVE="${NORM_PRESERVE:-true}"
SCALE="${SCALE:-1.0}"
SPARSITY="${SPARSITY:-0.0}"
DEST_LAYER_START="${DEST_LAYER_START:-0.30}"

# Resolve model source
if [ -n "$MODEL_PATH" ]; then
    MODEL="$MODEL_PATH"
elif [ -n "$MODEL_ID" ]; then
    MODEL="$MODEL_ID"
else
    echo "ERROR: Set MODEL_PATH (local HF dir) or MODEL_ID (HuggingFace repo ID)" >&2
    exit 1
fi

mkdir -p "$WORKSPACE_DIR" "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
run_measure() {
    echo "==> [measure] Collecting harmful/harmless activations"
    ARGS=(-m "$MODEL" -o "$MEASUREMENTS_FILE")
    [ -n "$QUANT" ]         && ARGS+=(--quant "$QUANT")
    [ -n "$DATA_HARMFUL" ]  && ARGS+=(--data-harmful "$DATA_HARMFUL")
    [ -n "$DATA_HARMLESS" ] && ARGS+=(--data-harmless "$DATA_HARMLESS")
    [ "$PROJECTED" = "true" ] && ARGS+=(--projected)
    cd /opt/abliterator/llm-abliteration
    python measure.py "${ARGS[@]}"
    echo "==> [measure] Done — measurements at $MEASUREMENTS_FILE"
}

run_analyze() {
    echo "==> [analyze] Layer-by-layer signal quality (stdout only)"
    cd /opt/abliterator/llm-abliteration
    python analyze.py "$MEASUREMENTS_FILE"
    echo "==> [analyze] Done"
}

gen_yaml() {
    echo "==> [auto_yaml] Generating ablation YAML from measurements"
    python /opt/abliterator/auto_yaml.py \
        --measurements "$MEASUREMENTS_FILE" \
        --model "$MODEL" \
        --output-dir "$OUTPUT_DIR" \
        --yaml-out "$ABLATION_YAML" \
        --scale "$SCALE" \
        --sparsity "$SPARSITY" \
        --dest-start "$DEST_LAYER_START"
    echo "==> [auto_yaml] YAML written to $ABLATION_YAML"
}

run_ablate() {
    echo "==> [ablate] Abliterating model weights"
    ARGS=("$ABLATION_YAML")
    [ "$NORM_PRESERVE" = "true" ] && ARGS+=(--normpreserve)
    [ "$PROJECTED" = "true" ]     && ARGS+=(--projected)
    cd /opt/abliterator/llm-abliteration
    python sharded_ablate.py "${ARGS[@]}"
    echo "==> [ablate] Done — model saved to $OUTPUT_DIR"
}

# ---------------------------------------------------------------------------
case "$MODE" in
    measure)
        run_measure
        ;;
    analyze)
        [ -f "$MEASUREMENTS_FILE" ] || { echo "ERROR: $MEASUREMENTS_FILE not found" >&2; exit 1; }
        run_analyze
        ;;
    ablate)
        [ -f "$MEASUREMENTS_FILE" ] || { echo "ERROR: $MEASUREMENTS_FILE not found" >&2; exit 1; }
        if [ ! -f "$ABLATION_YAML" ]; then
            gen_yaml
        else
            echo "==> Using existing ABLATION_YAML at $ABLATION_YAML"
        fi
        run_ablate
        ;;
    full)
        run_measure
        run_analyze
        gen_yaml
        run_ablate
        ;;
    *)
        echo "ERROR: Unknown MODE='$MODE'. Valid: measure | analyze | ablate | full" >&2
        exit 1
        ;;
esac
