import os
from pathlib import Path

import torch


def save_checkpoint(state, filename):
    """
    Write a checkpoint atomically.

    Saves to a sibling temporary file and renames it into place, so a
    run killed mid-write (a Colab disconnect while flushing to
    mounted Drive) leaves the previous checkpoint intact instead of a
    truncated file that cannot be resumed from.
    """

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_filename = filename.with_name(filename.name + ".tmp")

    torch.save(state, tmp_filename)

    os.replace(tmp_filename, filename)


def load_checkpoint(
    filename,
    device="cpu",
):

    filename = Path(filename)

    if not filename.exists():

        raise FileNotFoundError(filename)

    # weights_only=False: our checkpoints carry RNG snapshots and
    # training history, not just tensors. Passing it explicitly keeps
    # loading working on torch>=2.6, where the default flipped to True.
    return torch.load(
        filename,
        map_location=device,
        weights_only=False,
    )
