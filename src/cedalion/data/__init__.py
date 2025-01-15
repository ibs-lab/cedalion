from importlib.resources import files
from pathlib import Path


def get(fname: str | Path) -> Path:
    """Returns the absolute path to a datafile."""

    return files("cedalion.data").joinpath(fname)
