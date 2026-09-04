"""Public inference API: load a released LoQI checkpoint and generate conformers for SMILES."""

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

# Reference molecule size (atoms) of the atom-aware batch sampler; see featurize.build_sampling_loader.
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
    """Path of an inference config shipped in ``loqi/configs`` (``loqi.yaml`` or ``loqi_flow.yaml``)."""
    return Path(str(files("loqi").joinpath("configs", name)))


def load_config(name_or_path: str | Path) -> DictConfig:
    """Load a bundled config by name or any config YAML by path.

    ``sample.node_distribution`` is always set to ``None``: the node-count prior is only used for
    de novo sampling, and conformer generation always supplies the molecular graph.
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
    """``device`` as a :class:`torch.device`; ``None`` selects CUDA when available, else CPU."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class LoadedModel:
    """A LoQI model ready for sampling, together with the config it was loaded with."""

    model: Graph3DInterpolantModel
    config: DictConfig
    name: str
    checkpoint: Path
    device: torch.device

    @property
    def default_steps(self) -> int:
        """Sampling steps the model was trained with (``interpolant.timesteps``)."""
        return int(self.config.interpolant.timesteps)

    @property
    def default_batch_size(self) -> int:
        """Reference batch size (molecules of ``TARGET_MOLECULE_SIZE`` atoms) from the config."""
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
    """Load a registered model (downloading it into the cache on first use) or a local checkpoint.

    ``config`` is a bundled config name or a YAML path. It defaults to the registry entry's config;
    for local checkpoints whose file stem is not a registry name it must be given explicitly.
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
    # The checkpoint pickles omegaconf and megalodon objects. Lightning >= 2.5.5 exposes
    # torch.load's ``weights_only``; older versions load with weights_only=False internally.
    if "weights_only" in inspect.signature(Graph3DInterpolantModel.load_from_checkpoint).parameters:
        kwargs["weights_only"] = False
    model = Graph3DInterpolantModel.load_from_checkpoint(str(ckpt), map_location=dev, **kwargs)
    model.batch_preprocessor = preprocessor
    model = model.to(dev).eval()
    return LoadedModel(model=model, config=cfg, name=name, checkpoint=Path(ckpt), device=dev)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def iter_sampled_batches(
    loaded: LoadedModel, loader: DataLoader, *, steps: int | None = None
) -> Iterator[tuple[Batch, list[np.ndarray]]]:
    """Run the sampler over ``loader`` and yield ``(batch, coordinates)`` per batch.

    ``coordinates`` holds one ``(n_atoms, 3)`` array per molecule in the batch, in batch order.
    ``steps`` defaults to the number of steps the model was trained with.
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
    """Generate conformers for one SMILES string or a sequence of SMILES strings.

    Returns one RDKit molecule per input SMILES, in input order. Each molecule has explicit
    hydrogens (unless ``add_hs`` is False) and up to ``n_conformers`` conformers with ids
    ``0..k-1``. Samples with non-finite coordinates are dropped; their count is stored in the
    integer property ``loqi_failed`` on the molecule (``0`` when all samples succeeded).

    ``model`` is a registry name (``"loqi"``, ``"loqi_flow"``), a checkpoint path, or a
    :class:`LoadedModel` (``device`` is only used when a model has to be loaded). ``steps``
    defaults to the training step count (25); the diffusion model is not expected to work
    well with other values, the flow-matching model tolerates them. ``batch_atoms`` is the
    atom budget of a sampling batch at the 50-atom reference size (default:
    ``data.inference_batch_size * 50`` from the config, 7500 for the released models); batches
    of smaller molecules hold more atoms and batches of larger molecules fewer, so that the
    number of edges of the fully connected graphs stays roughly constant. Lower it to reduce
    memory use. Invalid SMILES raise :class:`ValueError` before any sampling happens.
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
