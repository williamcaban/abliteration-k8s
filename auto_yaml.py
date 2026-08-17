#!/usr/bin/env python3
"""
Auto-generate a sharded_ablate.py YAML config from a measurements .pt file.

Selects the measurement layer with the highest signal quality
(snr * (1 - cosine_similarity) * purity_ratio) in the middle-to-late
layer range, then targets all destination layers from dest_start onwards.
"""
import argparse
import torch
import torch.nn.functional as F
import yaml


def compute_signal_quality(results, layer):
    harmful = results[f"harmful_{layer}"].float()
    harmless = results[f"harmless_{layer}"].float()
    refusal = results[f"refuse_{layer}"].float()

    cos_sim = F.cosine_similarity(harmful, harmless, dim=0).item()
    harmful_norm = harmful.norm().item()
    harmless_norm = harmless.norm().item()
    refusal_norm = refusal.norm().item()

    snr = refusal_norm / max(harmful_norm, harmless_norm, 1e-8)

    harmless_unit = harmless / harmless.norm().clamp(min=1e-8)
    projection = (refusal @ harmless_unit) * harmless_unit
    refusal_orth = refusal - projection
    purity = refusal_orth.norm() / refusal.norm().clamp(min=1e-8)

    return float(snr * (1.0 - cos_sim) * purity)


def best_measurement_layer(results, n_layers, search_start=0.30, search_end=0.85):
    start = max(1, int(n_layers * search_start))
    end = min(n_layers - 1, int(n_layers * search_end))
    best_layer, best_q = start, -1.0
    for layer in range(start, end + 1):
        q = compute_signal_quality(results, layer)
        if q > best_q:
            best_q, best_layer = q, layer
    return best_layer, best_q


def generate_yaml(measurements_file, model, output_dir, yaml_out,
                  dest_start=0.30, scale=1.0, sparsity=0.0):
    print(f"Loading measurements: {measurements_file}")
    results = torch.load(measurements_file, map_location="cpu")
    n_layers = results["layers"]
    print(f"Total layers: {n_layers}")

    best_layer, quality = best_measurement_layer(results, n_layers)
    print(f"Best measurement layer: {best_layer}  (signal quality: {quality:.4f})")

    dest_start_idx = max(1, int(n_layers * dest_start))
    ablate_entries = [
        {"layer": l, "measurement": best_layer, "scale": scale, "sparsity": sparsity}
        for l in range(dest_start_idx, n_layers)
    ]

    config = {
        "model": model,
        "measurements": measurements_file,
        "output": output_dir,
        "ablate": ablate_entries,
    }

    with open(yaml_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"YAML written: {yaml_out}")
    print(f"Destination layers: {dest_start_idx}–{n_layers - 1} "
          f"({len(ablate_entries)} layers), measurement source: layer {best_layer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="Path to .pt measurements file")
    parser.add_argument("--model", required=True, help="HF model ID or local path (written into YAML)")
    parser.add_argument("--output-dir", required=True, help="Where sharded_ablate.py writes the model")
    parser.add_argument("--yaml-out", required=True, help="Path to write the generated YAML")
    parser.add_argument("--scale", type=float, default=1.0, help="Ablation scale factor (default: 1.0)")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity fraction (default: 0.0)")
    parser.add_argument("--dest-start", type=float, default=0.30,
                        help="Fraction of total layers to start ablating from (default: 0.30)")
    args = parser.parse_args()

    generate_yaml(
        args.measurements, args.model, args.output_dir, args.yaml_out,
        dest_start=args.dest_start, scale=args.scale, sparsity=args.sparsity,
    )
