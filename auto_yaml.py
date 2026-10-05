#!/usr/bin/env python3
"""
Auto-generate a sharded_ablate.py YAML config from a measurements .pt file.

Selects the measurement layer with the highest signal quality
(snr * (1 - cosine_similarity) * purity_ratio), then targets destination
layers from dest_start onwards.

Improvements over the original version:

1. Depth-scaled search window. The search window for the best measurement
   layer scales with network depth (start at depth*0.30 + 1, end at
   depth*0.70), matching the practice in the upstream reference config
   (gemma3-12b-it.yml measures from ~39% to ~69% of depth). The old fixed
   0.30-0.85 window over-weighted very late layers, which mostly carry
   token-binding features rather than refusal features.

2. Per-destination-layer measurement source (optional, --per-layer-source).
   Instead of one measurement layer for every destination, each destination
   layer ℓ uses the highest-quality measurement layer within a local window
   around it. The reference config does exactly this (layers 11-23 measured
   at 23, layers 24-41 measured at 29). Local sourcing tracks the refusal
   direction as it drifts across depth, which directly addresses the
   "partial compliance" artifact of single-source ablation: refusal is a
   manifold spread across layers (Wollschläger et al. 2025; pralab AAAI),
   so ablating one layer's direction leaves other layers' directions intact.

3. Destination-window bounds relative to depth (--dest-start / --dest-end as
   fractions), instead of always ablating to the last layer. Ablating the
   final layers degrades output quality (token binding) without removing
   much refusal.
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


def search_bounds(n_layers, start_frac, end_frac):
    start = max(1, int(n_layers * start_frac))
    end = min(n_layers - 1, int(n_layers * end_frac))
    return start, end


def best_measurement_layer(results, n_layers, search_start=0.30, search_end=0.70):
    """Highest signal-quality layer inside a depth-scaled window.

    Defaults scale with network depth: 0.30*depth to 0.70*depth — the
    middle-to-late band where refusal directions concentrate, excluding the
    late tail that carries token-binding features.
    """
    start, end = search_bounds(n_layers, search_start, search_end)
    best_layer, best_q = start, -1.0
    for layer in range(start, end + 1):
        q = compute_signal_quality(results, layer)
        if q > best_q:
            best_q, best_layer = q, layer
    return best_layer, best_q


def local_source_layer(results, n_layers, dest, window):
    """Highest-quality measurement layer within ±window of dest.

    Clamped to the searched band [1, n_layers-2]. Used when
    --per-layer-source is set: each destination layer is measured by the
    nearest high-quality layer instead of a single global source.
    """
    lo = max(1, dest - window)
    hi = min(n_layers - 2, dest + window)
    best_layer, best_q = lo, -1.0
    for layer in range(lo, hi + 1):
        q = compute_signal_quality(results, layer)
        if q > best_q:
            best_q, best_layer = q, layer
    return best_layer, best_q


def generate_yaml(measurements_file, model, output_dir, yaml_out,
                  dest_start=0.40, dest_end=0.90, scale=1.0, sparsity=0.0,
                  per_layer_source=False, source_window=6, search_start=0.30,
                  search_end=0.70):
    print(f"Loading measurements: {measurements_file}")
    results = torch.load(measurements_file, map_location="cpu")
    n_layers = results["layers"]
    print(f"Total layers: {n_layers}")

    best_layer, quality = best_measurement_layer(
        results, n_layers, search_start=search_start, search_end=search_end
    )
    print(f"Best measurement layer: {best_layer}  (signal quality: {quality:.4f})")

    dest_start_idx = max(1, int(n_layers * dest_start))
    dest_end_idx = min(n_layers - 1, int(n_layers * dest_end))

    ablate_entries = []
    for l in range(dest_start_idx, dest_end_idx + 1):
        if per_layer_source:
            src, src_q = local_source_layer(results, n_layers, l, source_window)
            if src != best_layer:
                print(f"  layer {l:3d}: local source {src} (q={src_q:.4f})")
        else:
            src, src_q = best_layer, quality
        ablate_entries.append(
            {"layer": l, "measurement": src, "scale": scale, "sparsity": sparsity}
        )

    config = {
        "model": model,
        "measurements": measurements_file,
        "output": output_dir,
        "ablate": ablate_entries,
    }

    with open(yaml_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"YAML written: {yaml_out}")
    print(f"Destination layers: {dest_start_idx}–{dest_end_idx} "
          f"({len(ablate_entries)} layers), "
          f"measurement source: {'per-layer (±%d)' % source_window if per_layer_source else f'layer {best_layer}'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", required=True, help="Path to .pt measurements file")
    parser.add_argument("--model", required=True, help="HF model ID or local path (written into YAML)")
    parser.add_argument("--output-dir", required=True, help="Where sharded_ablate.py writes the model")
    parser.add_argument("--yaml-out", required=True, help="Path to write the generated YAML")
    parser.add_argument("--scale", type=float, default=1.0, help="Ablation scale factor (default: 1.0)")
    parser.add_argument("--sparsity", type=float, default=0.0, help="Sparsity fraction (default: 0.0)")
    parser.add_argument("--dest-start", type=float, default=0.40,
                        help="Fraction of total layers to start ablating from (default: 0.40)")
    parser.add_argument("--dest-end", type=float, default=0.90,
                        help="Fraction of total layers to stop ablating at (default: 0.90)")
    parser.add_argument("--per-layer-source", action="store_true",
                        help="Each destination layer uses the best measurement layer within its "
                             "local window instead of one global source (matches the reference "
                             "config's split measurement sources)")
    parser.add_argument("--source-window", type=int, default=6,
                        help="Half-width of the local source window in layers (default: 6)")
    parser.add_argument("--search-start", type=float, default=0.30,
                        help="Fraction of depth to start searching for the best layer (default: 0.30)")
    parser.add_argument("--search-end", type=float, default=0.70,
                        help="Fraction of depth to end the search (default: 0.70)")
    args = parser.parse_args()

    generate_yaml(
        args.measurements, args.model, args.output_dir, args.yaml_out,
        dest_start=args.dest_start, dest_end=args.dest_end, scale=args.scale,
        sparsity=args.sparsity, per_layer_source=args.per_layer_source,
        source_window=args.source_window, search_start=args.search_start,
        search_end=args.search_end,
    )
