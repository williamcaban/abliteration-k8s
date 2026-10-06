"""Hybrid-architecture sharded ablation for Qwen3.5 (Gated DeltaNet + attention).

Upstream llm-abliteration's sharded_ablate.py only targets classic-transformer
keys (self_attn.o_proj / mlp.down_proj). Qwen3.5 uses a hybrid 3:1 stack
(linear_attention / full_attention) where:

  - full-attention layers carry self_attn.o_proj.weight  (6 of 24 on the 2B)
  - linear-attention (DeltaNet) layers carry linear_attn.out_proj.weight
    (the DeltaNet output projection that writes to the residual stream)
  - ALL layers carry mlp.down_proj.weight

This script extends the upstream YAML flow with linear_attn.out_proj targets so
every destination layer gets an output-projection edit. Everything else mirrors
upstream behavior: same safetensors transposition convention, same norm-preserve
and projected-orthogonalization math, same shard-by-shard processing.

Usage: python hybrid_ablate.py CONFIG.yml [--normpreserve] [--projected]
"""

import argparse
import gc
import json
import os
import shutil

import torch
import yaml
from pathlib import Path
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils import cached_file

# ---------------------------------------------------------------------
# Weight-modification math — identical to upstream sharded_ablate.py
# (PyTorch nn.Linear stores [out, in]; safetensors stores transposed)
# ---------------------------------------------------------------------


def modify_tensor(W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    original_dtype = W.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)
        projection = torch.matmul(W_gpu, refusal_normalized)
        W_gpu -= scale_factor * torch.outer(projection, refusal_normalized)
        result = W_gpu.T.to("cpu", dtype=original_dtype, non_blocking=True)
        del W_gpu, refusal_dir_gpu, refusal_normalized, projection
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return result.detach().clone()


def modify_tensor_norm_preserved(W: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
    original_dtype = W.dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        W_gpu = W.to(device, dtype=torch.float32, non_blocking=True).T
        refusal_dir_gpu = refusal_dir.to(device, dtype=torch.float32, non_blocking=True)
        if refusal_dir_gpu.dim() > 1:
            refusal_dir_gpu = refusal_dir_gpu.view(-1)
        refusal_normalized = torch.nn.functional.normalize(refusal_dir_gpu, dim=0)

        W_norm = torch.norm(W_gpu, dim=1, keepdim=True)
        W_direction = torch.nn.functional.normalize(W_gpu, dim=1)
        projection = torch.matmul(W_direction, refusal_normalized)
        W_direction_new = W_direction - scale_factor * torch.outer(projection, refusal_normalized)
        W_direction_new = torch.nn.functional.normalize(W_direction_new, dim=1)
        W_modified = W_norm * W_direction_new

        result = W_modified.T.to("cpu", dtype=original_dtype, non_blocking=True)
        del W_gpu, refusal_dir_gpu, refusal_normalized, projection
        del W_direction, W_direction_new, W_norm, W_modified
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    return result.detach().clone()


def magnitude_sparsify(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    if fraction >= 1.0:
        return tensor
    k = int(tensor.numel() * fraction)
    if k == 0:
        return torch.zeros_like(tensor)
    flat = tensor.flatten()
    threshold = torch.topk(flat.abs(), k, largest=True, sorted=False)[0].min()
    mask = tensor.abs() >= threshold
    return tensor * mask


# ---------------------------------------------------------------------
# Target-module resolution for hybrid architectures
# ---------------------------------------------------------------------

# Key suffixes that write to the residual stream (output projections), per
# layer type. Order matters: the first suffix present in the shard's weight
# map for a layer wins.
OUTPUT_PROJ_SUFFIXES = [
    ".self_attn.o_proj.weight",     # full-attention layers
    ".linear_attn.out_proj.weight",  # Gated DeltaNet (linear-attention) layers
    ".mlp.down_proj.weight",        # all layers
]


def build_layer_targets(weight_map: dict) -> tuple[str, dict[int, list[str]]]:
    """Return (layer_prefix, {layer_idx: [full key names]}).

    Detects the layer prefix from any per-layer key, then maps every layer to
    the output-projection keys that actually exist for it (self_attn.o_proj,
    linear_attn.out_proj, mlp.down_proj).
    """
    layer_prefix = None
    for key in weight_map:
        if ".layers." in key and (".self_attn." in key or ".linear_attn." in key or ".mlp." in key):
            layer_prefix = key.split(".layers.")[0]
            break
    if layer_prefix is None:
        raise ValueError("Could not detect layer structure in model weights")
    print(f"Detected layer prefix: {layer_prefix}")

    layer_keys: dict[int, list[str]] = {}
    for key in weight_map:
        if not key.startswith(f"{layer_prefix}.layers."):
            continue
        rest = key[len(f"{layer_prefix}.layers."):]
        head = rest.split(".", 1)[0]
        try:
            idx = int(head)
        except ValueError:
            continue
        if any(key.endswith(s) for s in OUTPUT_PROJ_SUFFIXES):
            layer_keys.setdefault(idx, []).append(key)
    return layer_prefix, layer_keys


# ---------------------------------------------------------------------
# Sharded ablation — mirrors upstream, with hybrid target resolution
# ---------------------------------------------------------------------


def ablate_by_layers_sharded(
    model_name: str,
    measures: dict,
    marching_orders: list,
    output_path: str,
    norm_preserve: bool,
    projected: bool,
) -> None:
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)

    precision = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None) or torch.float32
    if isinstance(precision, str):
        precision = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            precision, torch.float32
        )
    print(f"Model precision: {precision}")

    index_path = cached_file(model_name, "model.safetensors.index.json")
    model_dir = Path(index_path).parent
    print(f"Model directory: {model_dir}")

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    _prefix, layer_keys = build_layer_targets(weight_map)

    n_attn = sum(1 for ks in layer_keys.values() for k in ks if ".o_proj." in k)
    n_delta = sum(1 for ks in layer_keys.values() for k in ks if ".out_proj." in k)
    n_mlp = sum(1 for ks in layer_keys.values() for k in ks if ".down_proj." in k)
    print(f"Hybrid targets: {n_attn} self_attn.o_proj, {n_delta} linear_attn.out_proj, {n_mlp} mlp.down_proj keys")

    shard_modifications: dict[str, list] = {}
    missing = []
    for layer, measurement, scale, sparsity in marching_orders:
        keys = layer_keys.get(layer, [])
        if not keys:
            missing.append(layer)
            continue
        for key in keys:
            shard_file = weight_map[key]
            shard_modifications.setdefault(shard_file, []).append((key, layer, measurement, scale, sparsity))
    if missing:
        print(f"WARNING: no output-projection keys found for layers {missing}")

    print(f"\nWill modify {len(shard_modifications)} shards out of {len(set(weight_map.values()))} total")

    os.makedirs(output_path, exist_ok=True)

    all_shards = sorted(set(weight_map.values()))
    for shard_file in tqdm(all_shards, desc="Processing shards"):
        shard_path = model_dir / shard_file
        if shard_file in shard_modifications:
            print(f"\nLoading and modifying {shard_file}...")
            state_dict = load_file(str(shard_path))

            for key, layer, measurement, scale, sparsity in shard_modifications[shard_file]:
                if key not in state_dict:
                    continue
                print(f"  Modifying layer {layer}: {key}")

                refusal_dir = measures[f"refuse_{measurement}"].float()
                harmless_dir = measures[f"harmless_{layer}"].float()

                if projected:
                    harmless_normalized = torch.nn.functional.normalize(harmless_dir, dim=0)
                    projection_scalar = refusal_dir @ harmless_normalized
                    refined = refusal_dir - projection_scalar * harmless_normalized
                    refusal_dir = refined.to(precision)
                    del harmless_normalized, refined

                if sparsity > 0.0:
                    refusal_dir = magnitude_sparsify(refusal_dir, fraction=sparsity)

                refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=-1)

                if norm_preserve:
                    state_dict[key] = modify_tensor_norm_preserved(state_dict[key], refusal_dir, scale).contiguous()
                else:
                    state_dict[key] = modify_tensor(state_dict[key], refusal_dir, scale).contiguous()

                del refusal_dir, harmless_dir
                gc.collect()

            print(f"  Saving {shard_file}...")
            save_file(state_dict, f"{output_path}/{shard_file}")
            del state_dict
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            shutil.copy(str(shard_path), f"{output_path}/{shard_file}")

    print("\nCopying configuration files...")
    shutil.copy(str(index_path), f"{output_path}/model.safetensors.index.json")
    for file in [
        "config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
        "generation_config.json", "tokenizer.model", "vocab.json", "merges.txt",
        "added_tokens.json", "preprocessor_config.json", "chat_template.json",
    ]:
        try:
            src_path = cached_file(model_name, file)
            if src_path and os.path.exists(src_path):
                shutil.copy(src_path, f"{output_path}/{file}")
        except Exception:
            pass

    print(f"\nModified model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_path", type=str, help="Path to a YAML configuration file")
    parser.add_argument("--normpreserve", action="store_true", default=False, help="Preserve norms/magnitudes when ablating refusal")
    parser.add_argument("--projected", action="store_true", default=False, help="Project refusal against harmless direction and orthogonalize")

    args = parser.parse_args()

    with open(args.file_path, "r") as file:
        ydata = yaml.safe_load(file)

    model_name = ydata.get("model")
    measurement_file = ydata.get("measurements")
    output_dir = ydata.get("output")
    ablations = ydata.get("ablate")

    print("=" * 60)
    print("HYBRID SHARDED ABLATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Measurements: {measurement_file}")
    print(f"Output directory: {output_dir}")
    print(f"Number of ablations: {len(ablations)}")
    print(f"Norm preservation: {args.normpreserve}")
    print(f"Projected: {args.projected}")
    print("=" * 60)

    print(f"\nLoading measurements from {measurement_file}...")
    measures = torch.load(measurement_file)
    print(f"Loaded {len(measures)} measurements")

    orders = [
        (int(item["layer"]), int(item["measurement"]), float(item["scale"]), float(item["sparsity"]))
        for item in ablations
    ]

    print("\nAblation orders:")
    for layer, measurement, scale, sparsity in orders:
        print(f"  Layer {layer}: measurement={measurement}, scale={scale}, sparsity={sparsity}")

    print("\n" + "=" * 60)
    print("STARTING ABLATION")
    print("=" * 60)
    ablate_by_layers_sharded(
        model_name=model_name,
        measures=measures,
        marching_orders=orders,
        output_path=output_dir,
        norm_preserve=args.normpreserve,
        projected=args.projected,
    )
    print("\n" + "=" * 60)
    print("ABLATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
