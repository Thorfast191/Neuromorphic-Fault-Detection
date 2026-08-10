"""
Google Colab bootstrap.

Colab runtimes are wiped on disconnect, so anything worth keeping —
checkpoints, the cached split, logs — has to be written to mounted
Google Drive instead of the container's local disk. `setup()` mounts
Drive, points the project's results directory at a folder there via
`NFD_RESULTS_DIR`, and is a no-op outside Colab so the same notebook
cell also runs unchanged on a local machine.

Typical use, as the first cell of a notebook in a fresh clone:

    from utils.colab import setup
    paths = setup()
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from utils.paths import ENV_RESULTS_DIR, project_root

DEFAULT_MOUNTPOINT = "/content/drive"
DEFAULT_PROJECT_DIR = "Neuromorphic-Fault-Detection"

# Not preinstalled in Colab; the rest of requirements.txt is.
COLAB_PACKAGES = {"snntorch": "snntorch>=0.9.1"}


def _default_config() -> dict:
    """
    Load `configs/default.yaml` if present, so reported paths match
    what the training code will resolve. Missing or unreadable config
    is not fatal - resolution falls back to the known locations.
    """

    config_path = project_root() / "configs" / "default.yaml"

    if not config_path.exists():
        return {}

    try:
        from utils.config import Config

        return Config(config_path).data or {}
    except Exception:  # noqa: BLE001 - config is advisory here
        return {}


def in_colab() -> bool:
    """
    True when running inside a Google Colab runtime.
    """

    return importlib.util.find_spec("google.colab") is not None


def install_missing(packages: dict[str, str] | None = None) -> list[str]:
    """
    Pip-install only the packages that fail to import.

    Args:
        packages: Mapping of import name -> pip requirement.

    Returns:
        The requirements that were installed.
    """

    packages = COLAB_PACKAGES if packages is None else packages

    missing = [
        requirement
        for module, requirement in packages.items()
        if importlib.util.find_spec(module) is None
    ]

    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
        )

    return missing


def mount_drive(mountpoint: str = DEFAULT_MOUNTPOINT) -> Path:
    """
    Mount Google Drive, reusing an existing mount if present.

    Returns:
        Path to `MyDrive` inside the mountpoint.
    """

    from google.colab import drive  # noqa: PLC0415 - Colab-only import

    my_drive = Path(mountpoint) / "MyDrive"

    if not my_drive.exists():
        drive.mount(mountpoint)

    if not my_drive.exists():
        raise RuntimeError(
            f"Drive mounted at {mountpoint} but {my_drive} is missing."
        )

    return my_drive


def setup(
    project_dir: str = DEFAULT_PROJECT_DIR,
    mountpoint: str = DEFAULT_MOUNTPOINT,
    trial: str | None = None,
    install_deps: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Prepare the runtime for training.

    In Colab: installs missing dependencies, mounts Drive, and sets
    `NFD_RESULTS_DIR` to `MyDrive/<project_dir>/results` (or
    `MyDrive/<project_dir>/<trial>/results` when `trial` is given) so
    checkpoints survive a disconnect. Elsewhere: reports the local
    paths and changes nothing.

    Args:
        project_dir: Folder under `MyDrive` holding this project's
            results. Keep it stable across sessions to resume.
        mountpoint: Where to mount Drive.
        trial: Subfolder isolating this run's checkpoints, logs and
            report from other trials, e.g. "trial_2". Give every
            trial (a retrain with different code, config or data) its
            own name - reusing one silently mixes its checkpoints and
            cached split with a previous, possibly incompatible run.
            Omit to write straight to the project's `results/`
            folder.
        install_deps: Install missing packages (Colab only).
        verbose: Print a summary of the resolved paths.

    Returns:
        Dict with `in_colab`, `results_dir`, `data_root`, `trial` and
        `drive_dir` (None outside Colab).
    """

    # Imported here so `resolve_*` observes any env var set below.
    from utils.paths import resolve_dataset_root, resolve_results_dir

    cfg = _default_config()

    running_in_colab = in_colab()
    drive_dir = None
    installed: list[str] = []

    if running_in_colab:

        if install_deps:
            installed = install_missing()

        drive_dir = mount_drive(mountpoint) / project_dir
        results_dir = (drive_dir / trial / "results") if trial else (drive_dir / "results")
        results_dir.mkdir(parents=True, exist_ok=True)

        os.environ[ENV_RESULTS_DIR] = str(results_dir)

    # Resolved with the project config so these are exactly the paths
    # a notebook cell will get, not a lookalike.
    results_dir = resolve_results_dir(cfg)

    try:
        data_root = resolve_dataset_root(cfg)
    except FileNotFoundError as exc:
        data_root = None
        data_error = exc
    else:
        data_error = None

    if verbose:

        print(f"environment : {'Colab' if running_in_colab else 'local'}")

        if installed:
            print(f"installed   : {', '.join(installed)}")

        print(f"project root: {project_root()}")
        if trial:
            print(f"trial       : {trial}")
        print(f"results dir : {results_dir}")
        print(f"data root   : {data_root if data_root else f'NOT FOUND - {data_error}'}")

        if running_in_colab:
            print(
                "\nCheckpoints go to Drive. Re-run this notebook after a "
                "disconnect and training resumes from the last epoch."
            )

    return {
        "in_colab": running_in_colab,
        "results_dir": results_dir,
        "data_root": data_root,
        "trial": trial,
        "drive_dir": drive_dir,
    }
