from pathlib import Path

import torch


def save_checkpoint(state, filename):

    filename = Path(filename)

    filename.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(state, filename)


def load_checkpoint(
    filename,
    device="cpu",
):

    filename = Path(filename)

    if not filename.exists():

        raise FileNotFoundError(filename)

    return torch.load(
        filename,
        map_location=device,
    )