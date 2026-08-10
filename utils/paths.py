"""
Path resolution that survives moving between machines.

The repo is developed locally but trained in Colab, where it is a
fresh `git clone` and the results directory must live on mounted
Google Drive to outlive the runtime. Both locations are therefore
resolved at runtime rather than trusted verbatim from the config:

- results dir : `NFD_RESULTS_DIR` env var  >  `results.output_dir`
- dataset root: `NFD_DATA_ROOT` env var    >  `dataset.root`  >  the
  first directory in `DATASET_FALLBACKS` that actually contains MAT
  files (a clone has the tracked `cwru/`, not the gitignored
  `data/raw/CWRU/`).

Relative paths are resolved against the project root, so notebooks
work regardless of the kernel's working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_RESULTS_DIR = "NFD_RESULTS_DIR"
ENV_DATA_ROOT = "NFD_DATA_ROOT"

# Checked in order when the configured dataset root is missing.
DATASET_FALLBACKS = ("cwru", "data/raw/CWRU", "data/CWRU")


def project_root() -> Path:
    """
    Repository root (the directory containing `configs/`).
    """

    return Path(__file__).resolve().parent.parent


def _absolute(path: str | Path) -> Path:
    """
    Resolve `path` against the project root unless already absolute
    or already present relative to the working directory.
    """

    path = Path(path).expanduser()

    if path.is_absolute() or path.exists():
        return path

    return project_root() / path


def _has_mat_files(path: Path) -> bool:

    return path.is_dir() and any(path.rglob("*.mat"))


def resolve_results_dir(cfg: dict | None = None) -> Path:
    """
    Directory for checkpoints, splits and logs.

    Point `NFD_RESULTS_DIR` at a Drive folder (see `utils.colab`) to
    keep results across Colab sessions. Created if missing.
    """

    override = os.environ.get(ENV_RESULTS_DIR)

    if override:
        results_dir = Path(override).expanduser()
    else:
        configured = (cfg or {}).get("results", {}).get("output_dir", "results")
        results_dir = _absolute(configured)

    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def resolve_dataset_root(cfg: dict | None = None) -> Path:
    """
    Directory holding the CWRU MAT files.

    Raises:
        FileNotFoundError: If no candidate directory contains MAT
            files, listing what was tried.
    """

    override = os.environ.get(ENV_DATA_ROOT)

    if override:
        root = Path(override).expanduser()

        if not _has_mat_files(root):
            raise FileNotFoundError(
                f"{ENV_DATA_ROOT}={root} contains no .mat files."
            )

        return root

    configured = (cfg or {}).get("dataset", {}).get("root")

    candidates = [configured, *DATASET_FALLBACKS] if configured else list(DATASET_FALLBACKS)

    tried = []

    for candidate in candidates:

        path = _absolute(candidate)
        tried.append(str(path))

        if _has_mat_files(path):
            return path

    raise FileNotFoundError(
        "No CWRU .mat files found. Tried:\n  " + "\n  ".join(tried)
    )
