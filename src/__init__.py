import os
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

os.environ.setdefault("UV_CACHE_DIR", str(_root / ".uv-cache"))
os.environ.setdefault("TMPDIR", str(_root / ".tmp"))

_hf_root = _root / ".hf"
os.environ.setdefault("HF_HOME", str(_hf_root))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_hf_root / "transformers"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf_root / "hub"))

for _p in (
    _root / ".uv-cache",
    _root / ".tmp",
    _hf_root,
    _hf_root / "hub",
    _hf_root / "transformers",
):
    _p.mkdir(parents=True, exist_ok=True)

__version__ = 1.0
__author__ = "kapustazh"

# __all__ = []
