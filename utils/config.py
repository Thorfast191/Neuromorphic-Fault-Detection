from pathlib import Path
import yaml


class Config:
    """
    YAML configuration wrapper.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        with self.path.open("r") as f:
            self.data = yaml.safe_load(f)

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, **kwargs):
        self.data.update(kwargs)

    def save(self, path=None):
        path = Path(path or self.path)

        with path.open("w") as f:
            yaml.safe_dump(
                self.data,
                f,
                sort_keys=False,
            )

    def __repr__(self):
        return f"Config({self.path})"