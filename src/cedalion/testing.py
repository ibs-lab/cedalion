"""Utilitiy functions for tests."""

import contextlib
from pathlib import Path
import tempfile
from collections.abc import Iterator


@contextlib.contextmanager
def temporary_filename(suffix: str = None) -> Iterator[Path]:
    """Context that creates a temporary file, returns its name and deletes it on exit.

    Using this context to create a temporay files works around the problem that on
    Windows an open temporary files may not be reopened again.

    Adapted from https://stackoverflow.com/a/57701186.

    Args:
      suffix: filename extension

    Yields:
      The path of the temporary file.
    """

    try:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_name = Path(f.name)
        f.close()
        yield tmp_name
    finally:
        tmp_name.unlink()
