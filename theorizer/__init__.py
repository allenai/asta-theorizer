# __init__.py
# A thin library around the Theorizer code in src/, for running the stages one at a time.

from __future__ import annotations

from . import Config
from . import Evaluate
from . import Legacy
from . import PaperImport
from . import Parallel
from . import Pipeline
from . import Project
from . import Symbolic


def __getattr__(name):
    # th.Export loads on first use, so `python -m theorizer.Export` runs it without importing it twice
    if (name == "Export"):
        import importlib
        return importlib.import_module(".Export", __name__)
    raise AttributeError("module 'theorizer' has no attribute " + repr(name))
