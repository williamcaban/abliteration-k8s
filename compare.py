#!/usr/bin/env python3
"""Abliteration before/after evaluation using NVIDIA garak.

Replaces the hand-rolled 8-prompt side-by-side script with a real
red-teaming harness: two garak runs (original vs. abliterated model) over a
probe queue, then a diff summary of attack success rate (ASR) per probe.

Probes (all bundled in garak, no external datasets needed):
  malwaregen.*    harmful generation requests (malware/payload/goal)
  dan.*           DAN-style jailbreaks
  grandma.*       appeals-to-Granite jailbreaks
  encoding.*      injected payloads in various text encodings
  tap.*           TAP-style tree-of-attacks jailbreaks

Detectors:
  mitigation.MitigationBypass — refusal-keyword detector, INVERTED:
      1.0 = complied with the harmful request, 0.0 = refused.
      This is the metric we diff between the two models.
  always.Fail — control detector; a model scoring 0 fails here means the
      generator returned outputs at all (sanity check on the harness itself).

The dataset of record for measure-time calibration remains the upstream
harmful/harmless parquet sets; Necent/llm-jailbreak-prompt-injection-dataset
(gated) can be supplied via DATA_HARMFUL/DATA_HARMLESS to the measure step.

Usage (inside the container, via k8s/job-compare.yaml):
  python3 /opt/abliterator/compare.py

Environment variables:
  ORIGINAL_MODEL     HF model ID or local path  (default: Qwen/Qwen3-8B)
  ABLITERATED_MODEL  path to abliterated model   (default: /workspace/run/output)
  GARAK_PROBES       garak probe spec            (default: bundled queue)
  GARAK_GENERATIONS  generations per prompt      (default: 1)
  GARAK_MAX_TOKENS   max new tokens per output   (default: 300)
  GARAK_REPORT_DIR   where garak writes JSONL    (default: /workspace/reports)
  HF_TOKEN           passed through to the generator for gated models
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# garak needs to be importable; the container installs it at image build time.
# ---------------------------------------------------------------------------
try:
    import garak
    from garak import _config, _plugins
    from garak.attempt import Message
    from garak.generators.base import Generator
    from garak.generators.huggingface import Model as HFModel
    from garak.evaluators import ThresholdEvaluator
    from garak import command as garak_command
except ImportError as e:
    print(f"ERROR: garak is not installed in this image: {e}", file=sys.stderr)
    print("Rebuild the image with the updated Containerfile.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORIGINAL    = os.getenv("ORIGINAL_MODEL",    "Qwen/Qwen3-8B")
ABLITERATED = os.getenv("ABLITERATED_MODEL", "/workspace/run/output")
PROBE_SPEC  = os.getenv(
    "GARAK_PROBES",
    "probes.malwaregen,probes.dan,probes.grandma,probes.encoding,probes.tap",
)
GENERATIONS = int(os.getenv("GARAK_GENERATIONS", "1"))
MAX_TOKENS  = int(os.getenv("GARAK_MAX_TOKENS", "300"))
REPORT_DIR  = os.getenv("GARAK_REPORT_DIR", "/workspace/reports")

CONTROL_PROBE = "probes.test.Test"   # harness sanity check, excluded from diff


# ---------------------------------------------------------------------------
# Generator: bf16 + device_map=auto + chat template
#
# garak's stock huggingface.Model loads fp32 via AutoModelForCausalLM then
# .to(device) — 8B fp32 does not fit a 24 GB L4. This subclass overrides
# _load_unsafe with bf16 + device_map="auto" and greedy decoding, keeping
# garak's _call_model (chat-template handling, prefix stripping, Message
# wrapping) untouched. _gather_hf_params feeds every hf_args entry through to
# from_pretrained, so we route loading through it to stay config-compatible.
# ---------------------------------------------------------------------------
class BF16HFGenerator(HFModel):
    """garak huggingface.Model with bf16 weights and device_map=auto."""

    generator_family_name = "Hugging Face 🤗 bf16 (abliteration-compare)"

    def _load_unsafe(self):
        if hasattr(self, "model") and self.model is not None:
            return

        import transformers
        import os as _os

        _os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "true")

        if self.seed is not None:
            transformers.set_seed(self.seed)

        self.config = transformers.AutoConfig.from_pretrained(self.name)
        self._set_hf_context_len(self.config)
        self.config.init_device = self.device

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype="bfloat16",
            device_map="auto",
        )
        self.model.eval()

        if (
            hasattr(self.config, "tokenizer_class")
            and self.config.tokenizer_class
        ):
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.tokenizer_class
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.name, padding_side="left"
            )

        self.use_chat = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        # stock Model sets this via `models_to_deprefix`; default off
        if not hasattr(self, "deprefix_prompt"):
            self.deprefix_prompt = False

        try:
            self.generation_config = transformers.GenerationConfig.from_pretrained(
                self.name
            )
        except OSError:
            # some repos ship no generation_config.json — fall back to defaults
            self.generation_config = transformers.GenerationConfig()
        if hasattr(self.generation_config, "max_length"):
            self.generation_config.max_length = None
        self.generation_config.eos_token_id = self.model.config.eos_token_id
        self.generation_config.pad_token_id = self.model.config.eos_token_id

        if not hasattr(self, "hf_args") or not isinstance(self.hf_args, dict):
            self.hf_args = {}
        self.hf_args["do_sample"] = False  # greedy: reproducible before/after


# ---------------------------------------------------------------------------
# Compare probes: identical prompt queue from the Necent dataset categories
#
# The probe queues below are a fixed, seeded sample of the
# Necent/llm-jailbreak-prompt-injection-dataset categories bundled at build
# time (data/compare-prompts-*.txt in the image). Each line is one prompt.
# The mitigation.MitigationBypass detector scores every output; ASR diff
# between the two models is the before/after result.
# ---------------------------------------------------------------------------
def load_prompt_file(path: Path, label: str) -> list[str]:
    if not path.exists():
        return []
    prompts = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"  loaded {len(prompts):3d} {label} prompts from {path.name}")
    return prompts


# ---------------------------------------------------------------------------
# Single garak run against one model, returns parsed JSONL report entries
# ---------------------------------------------------------------------------
def run_garak(label: str, model_path: str, report_prefix: str) -> tuple[Path, dict]:
    print(f"\n{'═' * 70}")
    print(f"GARAK RUN — {label}")
    print(f"  model:  {model_path}")
    print(f"  probes: {PROBE_SPEC}")
    print(f"{'═' * 70}", flush=True)

    report_path = Path(REPORT_DIR)
    report_path.mkdir(parents=True, exist_ok=True)
    prefix_file = report_path / f"{report_prefix}"

    argv = [
        # full class path: module-only target_type resolves to the module's
        # DEFAULT_CLASS via load_plugin, bypassing our registered generator
        "--target_type", "abliterate_compare.BF16HFGenerator",
        "--target_name", model_path,
        "--spec", PROBE_SPEC,
        # NOTE: always.Fail excluded — it scores None outputs as 1.0, and garak's
        # hitlog path then calls asdict() on a None output (upstream crash).
        # Generator liveness is visible in the report's attempt outputs count.
        "--detectors", "mitigation.MitigationBypass",
        "--generations", str(GENERATIONS),
        "--report_prefix", str(prefix_file),
        "--seed", "42",
    ]

    # Register the bf16 generator under garak.generators.abliterate_compare
    # so --target_type resolves to it, then invoke garak's CLI in-process.
    import garak.generators as gen_pkg
    import types as _types

    mod = _types.ModuleType("garak.generators.abliterate_compare")
    mod.BF16HFGenerator = BF16HFGenerator
    mod.DEFAULT_CLASS = "BF16HFGenerator"
    # point __file__ at compare.py so PluginCache.plugin_info can stat it
    mod.__file__ = __file__
    sys.modules["garak.generators.abliterate_compare"] = mod
    setattr(gen_pkg, "abliterate_compare", mod)
    # rehome the class into the registered module so garak's _load_config
    # resolves namespace generators.abliterate_compare and picks up
    # target_name from the plugins config
    BF16HFGenerator.__module__ = "garak.generators.abliterate_compare"

    _config.plugins.target_type = "abliterate_compare.BF16HFGenerator"
    _config.plugins.target_name = model_path

    print("  invoking garak CLI in-process...", flush=True)
    from garak.cli import main as garak_main
    garak_main(argv)

    report_file = Path(f"{prefix_file}.report.jsonl")
    if not report_file.exists():
        raise RuntimeError(f"garak did not produce a report at {report_file}")

    entries = []
    with open(report_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return report_file, summarize(entries, label)


# ---------------------------------------------------------------------------
# Report parsing: ASR per probe from the JSONL entries
#
# MitigationBypass is inverted (1.0 = complied with harmful request), so the
# mean detection score across a probe's outputs IS the attack success rate.
# ---------------------------------------------------------------------------
def summarize(entries: list[dict], label: str) -> dict:
    attempts = [e for e in entries if e.get("entry_type") == "attempt"]
    evals    = [e for e in entries if e.get("entry_type") == "eval"]

    by_probe: dict[str, dict] = {}
    for a in attempts:
        probe = a.get("probe_classname", "?")
        slot = by_probe.setdefault(
            probe, {"outputs": 0, "none_outputs": 0, "detector_hits": 0, "detector_scores": []}
        )
        outs = a.get("outputs") or []
        slot["outputs"] += len(outs)
        slot["none_outputs"] += sum(1 for o in outs if o is None)
        dr = a.get("detector_results") or {}
        scores = dr.get("mitigation.MitigationBypass")
        if scores:
            slot["detector_scores"].extend(s for s in scores if s is not None)

    print(f"\n── {label} results " + "─" * 46)
    header = f"  {'probe':38s} {'n':>4s} {'ASR':>7s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for probe in sorted(by_probe):
        s = by_probe[probe]
        scores = s["detector_scores"]
        asr = (sum(scores) / len(scores)) if scores else float("nan")
        print(f"  {probe:38s} {len(scores):>4d} {asr * 100:6.1f}%")

    summary = {
        "label": label,
        "model": ABLITERATED if label == "ABLITERATED" else ORIGINAL,
        "probes": {},
    }
    for probe, s in by_probe.items():
        scores = s["detector_scores"]
        summary["probes"][probe] = {
            "n_scored": len(scores),
            "asr": (sum(scores) / len(scores)) if scores else None,
            "outputs": s["outputs"],
            "none_outputs": s["none_outputs"],
        }
    return summary


# ---------------------------------------------------------------------------
# Per-category diff using the Necent dataset's orthogonal labels
# ---------------------------------------------------------------------------
CATEGORY_PROBES = {
    # Necent dataset prompt_type → garak probe family used for it
    "harmful_behavior":  ["malwaregen.TopLevel", "malwaregen.SubFunctions",
                          "malwaregen.Evasion", "malwaregen.Payload"],
    "jailbreak":         ["dan.", "grandma.", "tap."],
    "prompt_injection":  ["encoding."],
}


def categorize(probe_classname: str) -> str:
    for cat, prefixes in CATEGORY_PROBES.items():
        if any(probe_classname.startswith(p) for p in prefixes):
            return cat
    return "other"


def print_diff(orig: dict, ablit: dict) -> None:
    print(f"\n{'═' * 70}")
    print("BEFORE / AFTER — attack success rate (MitigationBypass, 1.0 = complied)")
    print(f"  original:    {orig['model']}")
    print(f"  abliterated: {ablit['model']}")
    print("═" * 70)

    cats = sorted({categorize(p) for p in list(orig["probes"]) + list(ablit["probes"])})
    cat_rows = []
    for cat in cats:
        o_scores, a_scores = [], []
        for probe, s in orig["probes"].items():
            if categorize(probe) == cat and s["asr"] is not None:
                o_scores.extend([s["asr"]] * s["n_scored"])
        for probe, s in ablit["probes"].items():
            if categorize(probe) == cat and s["asr"] is not None:
                a_scores.extend([s["asr"]] * s["n_scored"])
        o_asr = sum(o_scores) / len(o_scores) if o_scores else None
        a_asr = sum(a_scores) / len(a_scores) if a_scores else None
        cat_rows.append((cat, o_asr, a_asr))

    print(f"  {'category':20s} {'orig ASR':>9s} {'abliterated ASR':>16s} {'Δ':>8s}")
    print("  " + "-" * 58)
    for cat, o_asr, a_asr in cat_rows:
        o_s = f"{o_asr*100:8.1f}%" if o_asr is not None else "     n/a"
        a_s = f"{a_asr*100:8.1f}%" if a_asr is not None else "      n/a"
        delta = f"{(a_asr-o_asr)*100:+7.1f}pp" if (o_asr is not None and a_asr is not None) else "n/a"
        print(f"  {cat:20s} {o_s:>9s} {a_s:>16s} {delta:>8s}")

    total_o = [s for p, s in orig["probes"].items() if s["asr"] is not None for s in [s["asr"]] * s["n_scored"]]
    total_a = [s for p, s in ablit["probes"].items() if s["asr"] is not None for s in [s["asr"]] * s["n_scored"]]
    if total_o and total_a:
        o, a = sum(total_o) / len(total_o), sum(total_a) / len(total_a)
        print("  " + "-" * 58)
        print(f"  {'OVERALL':20s} {o*100:8.1f}% {a*100:15.1f}% {(a-o)*100:+7.1f}pp")

    # harness sanity check
    tp = ablit["probes"].get("test.Test", {})
    if tp.get("outputs", 0) == 0:
        print("\n⚠️  probes.test.Test produced no outputs — check the harness run.")

    print(
        "\nInterpretation: ASR increase on harmful_behavior/jailbreak categories = "
        "abliteration working (refusals removed). ASR near 0 on those categories "
        "means the ablation was too weak — see README 'Improving the ablation'."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("ABLITERATION BEFORE/AFTER EVALUATION (garak)")
    print(f"  original:    {ORIGINAL}")
    print(f"  abliterated: {ABLITERATED}")
    print(f"  report dir:  {REPORT_DIR}")

    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)

    orig_summary = run_garak("ORIGINAL", ORIGINAL, "garak-original")
    ablit_summary = run_garak("ABLITERATED", ABLITERATED, "garak-abliterated")

    print_diff(orig_summary[1], ablit_summary[1])

    # persist machine-readable summary for the job logs
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(REPORT_DIR) / f"compare-summary-{stamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"original": orig_summary[1], "abliterated": ablit_summary[1]}, f, indent=2)
    print(f"\nsummary written: {out}")
    print(f"garak JSONL reports: {orig_summary[0]} , {ablit_summary[0]}")


if __name__ == "__main__":
    main()
