"""
Project initialization for src.

This module sets local cache directories for uv and Hugging Face tooling.
Keep downloads and temporary files inside repository for reproducible runs.
Also helps sandboxed environments.

              .     .       .  .   . .   .   . .    +  .
        .     .  :     .    .. :. .___---------___.
             .  .   .    .  :.:. _".^ .^ ^.  '.. :"-_. .
          .  :       .  .  .:../:            . .^  :.:.
              .   . :: +. :.:/: .   .    .        . . .:\
       .  :    .     . _ :::/:               .  ^ .  . .:\
        .. . .   . - : :.:./.                        .  .:\
        .      .     . :..|:                    .  .  ^. .:|
          .       . : : ..||        .                . . !:|
        .     . . . ::. ::||                           . :)/
       .   .     : . : .:.|. ######              .#######::|
        :.. .  :-  : .:  ::|.#######           ..########:|
       .  .  .  ..  .  .. :\\ ########          :######## :/
        .        .+ :: : -.:\\. ########       . ########.:/
          .  .+   . . . . :.:\\. #######       #######..:/
            :: . . . . ::.:..:.\\           .   .   ..:/
         .   .   .  .. :  -::::.\\.       | |     . .:/
            .  :  .  .  .-:."":.\\            ..:/
       .      -.   . . . .: .:::.:.\\.           .:/
      .   .   .  :      : ....::_:..:\\   ___.  :/
         .   .  .   .:. .. .  .: :.:.:\\       :/
           +   .   .   : . ::. :.:. .:.|\\:..:/|
           .         +   .  .  ...:: ..|  --.:|
      .      . . .   .  .  . ... :..:.."(  ..)"
       .   .       .      :  .   .: ::/  .  .::\
"""

import os
from pathlib import Path


def init_runtime_dirs() -> None:
    """Initialize local cache directories for runtime.

    Do not call at import-time. Call from CLI entrypoint.
    """

    root: Path = Path(__file__).resolve().parent.parent

    _ = os.environ.setdefault("UV_CACHE_DIR", str(root / ".uv-cache"))
    _ = os.environ.setdefault("TMPDIR", str(root / ".tmp"))

    hf_root: Path = root / ".hf"
    _ = os.environ.setdefault("HF_HOME", str(hf_root))
    _ = os.environ.setdefault(
        "TRANSFORMERS_CACHE", str(hf_root / "transformers")
    )
    _ = os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_root / "hub"))

    for p in (
        root / ".uv-cache",
        root / ".tmp",
        hf_root,
        hf_root / "hub",
        hf_root / "transformers",
    ):
        p.mkdir(parents=True, exist_ok=True)
