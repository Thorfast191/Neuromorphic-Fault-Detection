"""
General helper functions.
"""

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist.

    Returns:
        Path object.
    """

    p = Path(path)

    p.mkdir(parents=True, exist_ok=True)

    return p