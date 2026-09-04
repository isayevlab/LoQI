"""LoQI: low-energy, stereochemistry-aware molecular conformer generation."""

from importlib.metadata import PackageNotFoundError, version

from loqi.api import LoadedModel, generate_conformers, load_model
from loqi.registry import MODELS, ModelEntry, checkpoint_path

try:
    __version__ = version("loqi")
except PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

__all__ = [
    "MODELS",
    "LoadedModel",
    "ModelEntry",
    "__version__",
    "checkpoint_path",
    "generate_conformers",
    "load_model",
]
