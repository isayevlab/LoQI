#!/usr/bin/env python3
"""Build the LoQI ChEMBL3D stereo training dataset.

For every ChEMBL3D ``mol_id``, this script selects the conformer with the
lowest absolute energy across all conformers and observed stereochemistry
classes. It combines that conformer's coordinates with its matching SDF
topology, infers stereochemistry from the selected 3D geometry, and writes the
standard LoQI train/validation/test PyG datasets and statistics.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from utils_data import (
    add_stereo_bonds,
    DEFAULT_CHARGES_DICT,
    FULL_ATOM_ENCODER,
    compute_all_statistics,
    raw_to_pyg,
    save_pyg_dataset,
    save_statistics,
)


SPLIT_NAMES = ("train", "val", "test")
STANDARD_SUFFIXES = (
    "h.pt",
    "n_h.pickle",
    "atom_types_h.npy",
    "bond_types_h.npy",
    "charges_h.npy",
    "charges_prior_h.npy",
    "valency_h.pickle",
    "smiles.pickle",
    "bond_lengths_h.pickle",
    "angles_h.pickle",
    "dihedrals_h.pickle",
    "is_aromatic_h.npy",
    "is_in_ring_h.npy",
    "hybridization_h.npy",
)


@dataclass(frozen=True)
class SelectedConformer:
    """A selected Zarr row and its provenance."""

    group: int
    row: int
    mol_id: str
    energy: float
    stereo_id: int


def normalize_mol_id(value) -> str:
    """Normalize fixed-width Zarr byte strings to topology SDF names."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace").rstrip(" \x00")
    return str(value).strip()


def lowest_energy_rows(group_number: int, group) -> list[SelectedConformer]:
    """Select one absolute-energy-minimum row for each contiguous ``mol_id``.

    ChEMBL3D's ``relative_energy`` resets for stereochemistry blocks and can
    consequently contain several zeros for one molecule. Absolute ``energy``
    is therefore used to choose one stereoconfiguration and conformer.
    """
    mol_ids = np.asarray(group["mol_id"][:])
    energies = np.asarray(group["energy"][:])
    stereo_ids = np.asarray(group["stereo_id"][:])

    if not (len(mol_ids) == len(energies) == len(stereo_ids)):
        raise ValueError(f"Group {group_number:03d} has misaligned arrays")
    if len(mol_ids) == 0:
        return []
    if not np.isfinite(energies).all():
        bad = int(np.count_nonzero(~np.isfinite(energies)))
        raise ValueError(f"Group {group_number:03d} contains {bad} non-finite energies")

    start_mask = np.empty(len(mol_ids), dtype=bool)
    start_mask[0] = True
    start_mask[1:] = mol_ids[1:] != mol_ids[:-1]
    starts = np.flatnonzero(start_mask)
    ends = np.append(starts[1:], len(mol_ids))

    block_ids = mol_ids[starts]
    if len(np.unique(block_ids)) != len(block_ids):
        raise ValueError(
            f"Group {group_number:03d} has non-contiguous rows for at least one mol_id"
        )

    selected = []
    for start, end in zip(starts, ends):
        row = int(start + np.argmin(energies[start:end]))
        selected.append(
            SelectedConformer(
                group=int(group_number),
                row=row,
                mol_id=normalize_mol_id(mol_ids[row]),
                energy=float(energies[row]),
                stereo_id=int(stereo_ids[row]),
            )
        )
    return selected


def split_conformers(
    records: Sequence[SelectedConformer],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[SelectedConformer]]:
    """Reproducibly split selected conformers using the project split policy."""
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("train_ratio and val_ratio must be between zero and one")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than one")
    if len(records) < 3:
        raise ValueError("At least three selected molecules are required for three splits")

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(records))
    n_train = int(train_ratio * len(records))
    n_val = int(val_ratio * len(records))
    if n_train == 0 or n_val == 0 or n_train + n_val == len(records):
        raise ValueError("Dataset is too small for the requested split ratios")

    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    return {
        name: [records[int(index)] for index in split_indices[name]]
        for name in SPLIT_NAMES
    }


def parse_groups(value: str | None) -> list[int] | None:
    """Parse comma-separated atom-count groups and inclusive ranges."""
    if value is None:
        return None
    groups: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending group range: {item}")
            groups.update(range(start, end + 1))
        else:
            groups.add(int(item))
    return sorted(groups)


def load_size_grouped_dataset(dataset_dir: Path):
    """Load ChEMBL3D's bundled, version-matched Zarr reader."""
    loader_path = dataset_dir / "scripts" / "sgdataset.py"
    if not loader_path.is_file():
        raise FileNotFoundError(f"ChEMBL3D loader not found: {loader_path}")
    spec = importlib.util.spec_from_file_location("chembl3d_sgdataset", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ChEMBL3D loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SizeGroupedDataset(
        str(dataset_dir / "zarr_database"),
        keys=["coord", "energy", "mol_id", "stereo_id"],
    )


def collect_selected_conformers(
    dataset,
    groups: Iterable[int],
    limit_molecules: int | None = None,
) -> list[SelectedConformer]:
    """Scan requested size groups and collect selected Zarr row references."""
    records: list[SelectedConformer] = []
    for group_number in tqdm(list(groups), desc="Selecting energy minima"):
        records.extend(lowest_energy_rows(group_number, dataset[group_number]))
        if limit_molecules is not None and len(records) >= limit_molecules:
            return records[:limit_molecules]
    return records


def _records_by_group(
    splits: Mapping[str, Sequence[SelectedConformer]],
) -> dict[int, dict[str, list[SelectedConformer]]]:
    grouped: dict[int, dict[str, list[SelectedConformer]]] = defaultdict(
        lambda: {name: [] for name in SPLIT_NAMES}
    )
    for split_name in SPLIT_NAMES:
        for record in splits[split_name]:
            grouped[record.group][split_name].append(record)
    return dict(grouped)


def _convert_record(topology: Chem.Mol, coords: np.ndarray, mol_id: str):
    """Convert one topology/coordinate pair using LoQI's existing helpers."""
    topology = Chem.Mol(topology)
    Chem.SanitizeMol(topology)
    Chem.Kekulize(topology, clearAromaticFlags=True)
    graph = raw_to_pyg(topology, coords)
    graph.chemblid = mol_id
    graph.edge_index, graph.edge_attr = add_stereo_bonds(
        graph.mol,
        chi_bonds=[7, 8],
        ez_bonds={Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6},
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        from_3D=True,
    )
    return graph


def _selected_coordinates(group, records: Sequence[SelectedConformer]) -> dict[int, np.ndarray]:
    """Read selected coordinates together to avoid repeated Zarr chunk decoding."""
    sorted_records = sorted(records, key=lambda record: record.row)
    rows = np.asarray([record.row for record in sorted_records], dtype=np.int64)
    coord_array = group["coord"]
    if hasattr(coord_array, "oindex"):
        coordinates = np.asarray(coord_array.oindex[rows])
    else:
        coordinates = np.asarray([coord_array[int(row)] for row in rows])
    return {record.row: coord for record, coord in zip(sorted_records, coordinates)}


def build_graph_splits(
    dataset,
    topology_folder: Path,
    splits: Mapping[str, Sequence[SelectedConformer]],
) -> tuple[dict[str, list], int, int]:
    """Stream topology SDFs once per group and build all three graph splits."""
    graphs = {name: [] for name in SPLIT_NAMES}
    grouped = _records_by_group(splits)
    missing_topologies = 0
    failed_conversions = 0

    for group_number in tqdm(sorted(grouped), desc="Building molecular graphs"):
        group_records = grouped[group_number]
        wanted = {
            record.mol_id: (split_name, order, record)
            for split_name in SPLIT_NAMES
            for order, record in enumerate(group_records[split_name])
        }
        all_records = [selection[2] for selection in wanted.values()]
        coordinates = _selected_coordinates(dataset[group_number], all_records)
        found: dict[str, list[tuple[int, object]]] = {
            name: [] for name in SPLIT_NAMES
        }
        seen_topologies: set[str] = set()
        sdf_path = topology_folder / f"{group_number:03d}.sdf"
        if not sdf_path.is_file():
            raise FileNotFoundError(f"Topology file not found: {sdf_path}")

        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
        for topology in supplier:
            if topology is None or not topology.HasProp("_Name"):
                continue
            mol_id = topology.GetProp("_Name").strip()
            selection = wanted.get(mol_id)
            if selection is None:
                continue
            seen_topologies.add(mol_id)
            split_name, order, record = selection
            try:
                graph = _convert_record(topology, coordinates[record.row], mol_id)
                found[split_name].append((order, graph))
            except Exception as exc:
                failed_conversions += 1
                print(f"Warning: failed to process {mol_id}: {exc}")

        missing = sorted(set(wanted) - seen_topologies)
        missing_topologies += len(missing)
        if missing:
            print(
                f"Warning: group {group_number:03d} is missing "
                f"{len(missing)} selected topologies"
            )

        for split_name in SPLIT_NAMES:
            found[split_name].sort(key=lambda item: item[0])
            graphs[split_name].extend(graph for _, graph in found[split_name])

    return graphs, missing_topologies, failed_conversions


def expected_output_paths(processed_path: Path) -> list[Path]:
    """Return the 42 standard artifacts (three splits times fourteen files)."""
    return [
        processed_path / f"{split_name}_{suffix}"
        for split_name in SPLIT_NAMES
        for suffix in STANDARD_SUFFIXES
    ]


def save_graph_splits(graphs: dict[str, list], processed_path: Path) -> None:
    """Write PyG datasets and all standard statistics for each split."""
    for split_name in SPLIT_NAMES:
        split_graphs = graphs[split_name]
        if not split_graphs:
            raise ValueError(f"No successfully converted molecules in {split_name} split")
        print(f"Saving {len(split_graphs)} {split_name} molecules")
        save_pyg_dataset(split_graphs, processed_path / f"{split_name}_h.pt")
        statistics = compute_all_statistics(
            split_graphs,
            FULL_ATOM_ENCODER,
            DEFAULT_CHARGES_DICT,
            additional_features=True,
        )
        save_statistics(statistics, processed_path, split_name)


def process(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    topology_folder = dataset_dir / "topologies"
    if not topology_folder.is_dir():
        raise FileNotFoundError(f"Topology directory not found: {topology_folder}")
    if args.limit_molecules < 0:
        raise ValueError("limit_molecules cannot be negative")

    dataset = load_size_grouped_dataset(dataset_dir)
    available_groups = dataset.keys()
    requested_groups = parse_groups(args.groups)
    if args.test_mode and requested_groups is None:
        requested_groups = [10]
    groups = available_groups if requested_groups is None else requested_groups
    unknown = sorted(set(groups) - set(available_groups))
    if unknown:
        raise ValueError(f"Groups absent from ChEMBL3D: {unknown}")

    limit_molecules = args.limit_molecules or None
    if args.test_mode and limit_molecules is None:
        limit_molecules = 30

    processed_path = Path(args.save_data_folder).expanduser().resolve() / "processed"
    existing = [path for path in expected_output_paths(processed_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {len(existing)} existing output files; pass --overwrite"
        )
    processed_path.mkdir(parents=True, exist_ok=True)

    records = collect_selected_conformers(dataset, groups, limit_molecules)
    print(f"Selected {len(records)} lowest-energy molecule records")
    splits = split_conformers(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Split sizes: " + ", ".join(f"{key}={len(value)}" for key, value in splits.items()))

    graphs, missing, failed = build_graph_splits(dataset, topology_folder, splits)
    save_graph_splits(graphs, processed_path)
    print(
        f"Completed: output={processed_path}, missing_topologies={missing}, "
        f"failed_conversions={failed}"
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and preprocess lowest-energy ChEMBL3D stereoconformers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="ChEMBL3D root containing zarr_database/, topologies/, and scripts/",
    )
    parser.add_argument(
        "--save_data_folder",
        required=True,
        help="Output dataset root; standard artifacts are written under processed/",
    )
    parser.add_argument(
        "--groups",
        default=None,
        help="Optional atom-count groups, e.g. '10,12-15'; default is every group",
    )
    parser.add_argument(
        "--limit_molecules",
        type=int,
        default=0,
        help="Optional total selected-molecule limit (0 means unlimited)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Smoke test group 010 with 30 molecules unless groups/limit are supplied",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of standard output files in the destination",
    )
    return parser


def main() -> None:
    process(setup_argument_parser().parse_args())


if __name__ == "__main__":
    main()
