from pathlib import Path
import json


def save_json(
    obj,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w") as f:
        json.dump(
            obj,
            f,
            indent=4,
        )


def load_json(path):

    with open(path) as f:

        return json.load(f)