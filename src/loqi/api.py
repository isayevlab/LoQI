"""Checkpoint loading and conformer sampling."""

from __future__ import annotations

import inspect
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from loqi import featurize
from loqi.registry import MODELS, checkpoint_path
from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.metrics.conformer_evaluation_callback import convert_coords_to_np
from megalodon.models.module import Graph3DInterpolantModel

TARGET_MOLECULE_SIZE = 50

__all__ = [
    "TARGET_MOLECULE_SIZE",
    "LoadedModel",
    "generate_conformers",
    "iter_sampled_batches",
    "load_config",
    "load_model",
    "resolve_device",
    "seed_everything",
]


def bundled_config_path(name: str) -> Path:
    """Locate a YAML configuration in the installed package."""
    return Path(str(files("loqi").joinpath("configs", name)))


def load_config(name_or_path: str | Path) -> DictConfig:
    """Read a bundled configuration or a YAML file.

    Disable the node-count prior because conformer sampling supplies a molecular graph.
    """
    path = Path(name_or_path)
    if not path.is_file():
        path = bundled_config_path(str(name_or_path))
    if not path.is_file():
        bundled = sorted(entry.config for entry in MODELS.values())
        raise FileNotFoundError(f"Config {str(name_or_path)!r} not found; pass a YAML path or one of {bundled}.")
    cfg = OmegaConf.load(path)
    OmegaConf.update(cfg, "sample.node_distribution", None, force_add=True)
    return cfg


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Use the requested device, or select CUDA when available and CPU otherwise."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class LoadedModel:
    """A loaded model with its configuration, checkpoint path, and device."""

    model: Graph3DInterpolantModel
    config: DictConfig
    name: str
    checkpoint: Path
    device: torch.device

    @property
    def default_steps(self) -> int:
        """Return the configured sampling step count."""
        return int(self.config.interpolant.timesteps)

    @property
    def default_batch_size(self) -> int:
        """Return the configured batch size at the reference molecule size."""
        data = self.config.data
        return int(data.get("inference_batch_size", data.get("batch_size", 32)))


def load_model(
    name_or_path: str | Path = "loqi",
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    config: str | Path | None = None,
    progress: bool = True,
) -> LoadedModel:
    """Load a registered model or a local checkpoint for inference.

    Registered checkpoints are downloaded and verified on first use. ``config`` may
    be a bundled name or a YAML path; it is required for unregistered checkpoint names.
    """
    name = str(name_or_path)
    ckpt = checkpoint_path(name, cache_dir=cache_dir, progress=progress)
    if config is None:
        key = name if name in MODELS else Path(name).stem
        if key not in MODELS:
            raise ValueError(
                f"No bundled config is associated with {name!r}; pass config='loqi.yaml', "
                "config='loqi_flow.yaml' or a path to a config YAML."
            )
        config = MODELS[key].config
    cfg = load_config(config)
    dev = resolve_device(device)

    preprocessor = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    kwargs = {
        "loss_params": cfg.loss,
        "interpolant_params": cfg.interpolant,
        "sampling_params": cfg.sample,
        "batch_preprocessor": preprocessor,
    }
    if "weights_only" in inspect.signature(Graph3DInterpolantModel.load_from_checkpoint).parameters:
        kwargs["weights_only"] = False
    model = Graph3DInterpolantModel.load_from_checkpoint(str(ckpt), map_location=dev, **kwargs)
    model.batch_preprocessor = preprocessor
    model = model.to(dev).eval()
    return LoadedModel(model=model, config=cfg, name=name, checkpoint=Path(ckpt), device=dev)


def seed_everything(seed: int) -> None:
    """Set the Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_sampled_batches(
    loaded: LoadedModel, loader: DataLoader, *, steps: int | None = None
) -> Iterator[tuple[Batch, list[np.ndarray]]]:
    """Yield each graph batch and its sampled coordinate arrays in batch order.

    Each molecule has an ``(n_atoms, 3)`` array. ``steps`` defaults to the model configuration.
    """
    steps = loaded.default_steps if steps is None else int(steps)
    model = loaded.model
    for batch in loader:
        batch = batch.to(model.device)
        sample = model.sample(batch=batch, timesteps=steps, pre_format=True)
        yield batch, convert_coords_to_np(sample)


def generate_conformers(
    smiles: str | Iterable[str],
    n_conformers: int = 10,
    *,
    model: str | Path | LoadedModel = "loqi",
    device: str | torch.device | None = None,
    seed: int = 42,
    steps: int | None = None,
    add_hs: bool = True,
    batch_atoms: int | None = None,
) -> list[Chem.Mol]:
    """Sample conformers and return one RDKit molecule per input SMILES, in input order.

    Each result contains up to ``n_conformers`` conformers with consecutive IDs.
    Non-finite samples are omitted and counted in the ``loqi_failed`` property.
    Hydrogens are explicit unless ``add_hs=False``. Invalid inputs raise ``ValueError``.

    ``model`` accepts a registry name, checkpoint path, or ``LoadedModel``. ``device``
    applies when loading a model. Reuse a loaded model for repeated calls.

    ``steps`` defaults to 25; use that value for diffusion checkpoints. ``batch_atoms``
    sets the atom budget at the 50-atom reference size (default 7500). Adaptive batching
    adjusts it for molecule size to keep graph edge counts roughly constant. Lower
    the budget to reduce memory use.
    """
    smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
    if not smiles_list:
        return []
    if n_conformers < 1:
        raise ValueError("n_conformers must be at least 1.")

    seed_everything(seed)
    loaded = model if isinstance(model, LoadedModel) else load_model(model, device=device)

    with featurize.legacy_stereo_perception():
        mols = [featurize.prepare_molecule(smi, add_hs=add_hs)[0] for smi in smiles_list]
        data_list = featurize.mols_to_data_list(mols, n_conformers, use_3d_input=False, use_stereo_bonds=True)

    if batch_atoms is None:
        reference_batch_size = loaded.default_batch_size
    else:
        reference_batch_size = max(1, int(batch_atoms) // TARGET_MOLECULE_SIZE)
    loader = featurize.build_sampling_loader(
        data_list,
        reference_batch_size,
        atom_aware_batching=True,
        shuffle=False,
        target_molecule_size=TARGET_MOLECULE_SIZE,
    )

    coords_per_mol: list[list[np.ndarray]] = [[] for _ in mols]
    n_failed = [0] * len(mols)
    for batch, coords_list in iter_sampled_batches(loaded, loader, steps=steps):
        for mol_idx, coords in zip(batch.mol_idx.tolist(), coords_list, strict=True):
            if np.isfinite(coords).all():
                coords_per_mol[mol_idx].append(coords)
            else:
                n_failed[mol_idx] += 1

    results = []
    for mol, coords, failed in zip(mols, coords_per_mol, n_failed, strict=True):
        out = featurize.conformers_to_mol(mol, coords)
        out.SetIntProp("loqi_failed", failed)
        results.append(out)
    return results
