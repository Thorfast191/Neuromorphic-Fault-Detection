"""
Signal loader.
"""

from pathlib import Path

from scipy.io import loadmat


class SignalLoader:

    def __call__(self, file):

        file = Path(file)

        mat = loadmat(file)

        for key in mat:

            if key.endswith("DE_time"):

                return mat[key].flatten()

        raise RuntimeError(
            "Drive-end signal not found."
        )