"""
Regenerates missing hf_model/ directories for pruned checkpoints.

For each checkpoint that has a pytorch_model.bin but no hf_model/config.json,
this script loads the checkpoint on CPU, extracts the model config and tokenizer,
and saves them to hf_model/. The weights themselves stay in pytorch_model.bin.

Usage:
  python regenerate_hf_model.py                       # all missing checkpoints
  python regenerate_hf_model.py prune_log/my_run      # single checkpoint dir
"""
import os
import sys
import glob
import torch
import torch.nn as nn

# ── Compatibility shims for checkpoints saved with older transformers ──────────
# 1. transformers removed several activation wrapper classes in 4.41+.
# 2. transformers.core_model_loading was an internal module removed in ~4.36.
# Re-register them so torch.load can unpickle old .bin files.

# --- Module-level shims (must come before any transformers import) ---
import types as _types

class _LazyStubModule(_types.ModuleType):
    """Auto-creates stub classes for any attribute access (used by pickle)."""
    def __getattr__(self, name):
        stub = type(name, (nn.Module,), {
            "__init__": lambda self, *a, **kw: nn.Module.__init__(self),
            "forward":  lambda self, x: x,
            "__module__": self.__name__,
        })
        setattr(self, name, stub)
        return stub

for _mod_name in [
    "transformers.core_model_loading",
    "transformers.modeling_utils_extra",   # just in case
]:
    if _mod_name not in sys.modules:
        _stub_mod = _LazyStubModule(_mod_name)
        sys.modules[_mod_name] = _stub_mod
        # Also attach as sub-attribute on the transformers package
        _parts = _mod_name.split(".")
        if len(_parts) == 2:
            import transformers as _tf
            if not hasattr(_tf, _parts[1]):
                setattr(_tf, _parts[1], _stub_mod)

import transformers.activations as _acts

def _make_shim(base_fn):
    """Create a minimal nn.Module wrapper around an activation function."""
    class _Shim(nn.Module):
        def forward(self, x):
            return base_fn(x)
    return _Shim

import torch.nn.functional as F

for _name, _fn in [
    ("SiLUActivation",    F.silu),
    ("GELUActivation",    F.gelu),
    ("NewGELUActivation", lambda x: F.gelu(x, approximate="tanh")),
    ("FastGELUActivation",lambda x: F.gelu(x, approximate="tanh")),
    ("QuickGELUActivation",lambda x: x * torch.sigmoid(1.702 * x)),
]:
    if not hasattr(_acts, _name):
        setattr(_acts, _name, _make_shim(_fn))
# ─────────────────────────────────────────────────────────────────────────────

def regenerate(ckpt_dir):
    bin_path = os.path.join(ckpt_dir, "pytorch_model.bin")
    hf_dir   = os.path.join(ckpt_dir, "hf_model")
    cfg_path = os.path.join(hf_dir, "config.json")

    if not os.path.exists(bin_path):
        print(f"  SKIP (no pytorch_model.bin): {ckpt_dir}")
        return
    if os.path.exists(cfg_path):
        print(f"  SKIP (already exists):       {ckpt_dir}")
        return

    print(f"  Generating hf_model/ for:    {ckpt_dir}")
    ckpt = torch.load(bin_path, map_location="cpu", weights_only=False)
    model     = ckpt["model"]
    tokenizer = ckpt["tokenizer"]

    os.makedirs(hf_dir, exist_ok=True)
    model.config.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    # generation_config if present
    if hasattr(model, "generation_config"):
        try:
            model.generation_config.save_pretrained(hf_dir)
        except Exception:
            pass

    del ckpt, model, tokenizer
    print(f"    -> saved to {hf_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dirs = sys.argv[1:]
    else:
        # Auto-discover all checkpoint dirs under prune_log/
        base = os.path.join(os.path.dirname(__file__), "prune_log")
        dirs = sorted(
            d for d in glob.glob(os.path.join(base, "*"))
            if os.path.isdir(d)
        )

    print(f"Checking {len(dirs)} checkpoint(s)...")
    for d in dirs:
        regenerate(d)
    print("Done.")
