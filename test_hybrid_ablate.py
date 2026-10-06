"""Unit tests for hybrid_ablate.build_layer_targets (target resolution only —
no torch needed; the module's torch import is mocked out).

Verifies hybrid target resolution against the REAL Qwen3.5-2B and Qwen3.5-4B
safetensors index structures fetched from the HF Hub (network required).
"""

import json
import sys
import types
import urllib.request
from pathlib import Path

import pytest

# hybrid_ablate imports torch/safetensors/transformers at module level; the
# test host may not have them. Stub them before the first import.
for name in ["torch", "safetensors", "safetensors.torch", "transformers",
             "transformers.utils", "yaml", "tqdm"]:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
sys.modules["safetensors.torch"].load_file = lambda *a, **k: {}
sys.modules["safetensors.torch"].save_file = lambda *a, **k: None
sys.modules["transformers.utils"].cached_file = lambda *a, **k: ""
sys.modules["transformers"].AutoConfig = type("AutoConfig", (), {})
sys.modules["tqdm"].tqdm = lambda *a, **k: ()
sys.modules["torch"].Tensor = type("Tensor", (), {"__module__": "torch"})


# Allow attribute assignment on the stub modules
class _StubModule(types.ModuleType):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_ablate import build_layer_targets  # noqa: E402

FULL_ATTN_INTERVAL = 4  # Qwen3.5: every 4th layer is full attention (3,7,11,...)


def _fetch_index(model_id: str) -> dict:
    url = f"https://huggingface.co/{model_id}/resolve/main/model.safetensors.index.json"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)["weight_map"]


def _check(wm: dict, n_layers: int, label: str):
    prefix, layer_keys = build_layer_targets(wm)
    assert prefix == "model.language_model", (label, prefix)
    # mtp.* (multi-token-prediction head) keys must not leak into layer keys
    assert all("mtp" not in k for ks in layer_keys.values() for k in ks)
    assert sorted(layer_keys) == list(range(n_layers)), (label, sorted(layer_keys))

    attn = sorted(l for l in layer_keys if any(".o_proj." in k for k in layer_keys[l]))
    delta = sorted(l for l in layer_keys if any(".out_proj." in k for k in layer_keys[l]))
    expected_attn = [l for l in range(n_layers) if l % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1]
    assert attn == expected_attn, (label, attn, expected_attn)
    # every layer must have at least one output-projection target
    assert sorted(set(attn) | set(delta)) == list(range(n_layers)), label
    # down_proj exists on ALL layers
    for l in range(n_layers):
        assert any(".mlp.down_proj." in k for k in layer_keys[l]), (label, l)
    return attn, delta


def test_qwen35_2b_targets():
    wm = _fetch_index("Qwen/Qwen3.5-2B")
    attn, delta = _check(wm, 24, "2B")
    assert attn == [3, 7, 11, 15, 19, 23]
    assert delta == [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]


def test_qwen35_4b_targets():
    wm = _fetch_index("Qwen/Qwen3.5-4B")
    attn, delta = _check(wm, 32, "4B")
    assert attn == [3, 7, 11, 15, 19, 23, 27, 31]


def test_classic_transformer_still_resolves():
    """A classic decoder-only map (Llama-style) must still resolve via o_proj."""
    wm = {f"model.layers.{l}.self_attn.o_proj.weight": "m.safetensors" for l in range(4)}
    wm.update({f"model.layers.{l}.mlp.down_proj.weight": "m.safetensors" for l in range(4)})
    prefix, lk = build_layer_targets(wm)
    assert prefix == "model"
    assert sorted(lk) == list(range(4))
    assert all(any(".o_proj." in k or ".down_proj." in k for k in lk[l]) for l in range(4))


def test_no_layer_structure_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        build_layer_targets({"model.embed_tokens.weight": "m.safetensors"})
