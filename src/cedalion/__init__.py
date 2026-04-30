from .physunits import Quantity, units

import cedalion.dataclasses
import cedalion.dataclasses.accessors
from cedalion.bibliography import Bibliography
from ._version import __version__


# singleton container that collects references
bib = Bibliography()

def cite(key: str) -> None:
    """Record that a method with this BibTeX key was used."""
    bib.cite(key)
