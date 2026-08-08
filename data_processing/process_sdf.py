#!/usr/bin/env python3
"""Convert a 3D SDF file to the standard LoQI training dataset format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from rdkit import Chem
from tqdm import tqdm

from process_chembl3d import (
    _convert_record,
    expected_output_paths,
    save_graph_splits,
    split_conformers,
)


def read_sdf_graphs(
    sdf_path: Path,
    limit_molecules: int | None = None,
) -> tuple[list, int]:
    """Read one 3D conformer per SDF record and convert it to a LoQI graph."""
    graphs = []
    failed_conversions = 0
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)

    records = enumerate(tqdm(supplier, desc="Converting SDF records"))
    for record_index, molecule in records:
        if molecule is None:
            failed_conversions += 1
            print(f"Warning: failed to read SDF record {record_index}")
            continue

        try:
            Chem.SanitizeMol(molecule)
            if molecule.GetNumConformers() != 1:
                raise ValueError(
                    f"expected exactly one conformer, found {molecule.GetNumConformers()}"
                )
            conformer = molecule.GetConformer()
            if not conformer.Is3D():
                raise ValueError("conformer is not marked as 3D")
            coordinates = np.asarray(conformer.GetPositions())
            if not np.isfinite(coordinates).all():
                raise ValueError("conformer contains non-finite coordinates")

            mol_id = (
                molecule.GetProp("_Name").strip()
                if molecule.HasProp("_Name") and molecule.GetProp("_Name").strip()
                else f"sdf_{record_index}"
            )
            graphs.append(_convert_record(molecule, coordinates, mol_id))
        except Exception as exc:
            failed_conversions += 1
            print(f"Warning: failed to process SDF record {record_index}: {exc}")

        if limit_molecules is not None and len(graphs) >= limit_molecules:
            break

    return graphs, failed_conversions


def process(args: argparse.Namespace) -> None:
    sdf_path = Path(args.sdf_path).expanduser().resolve()
    if not sdf_path.is_file():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")
    if args.limit_molecules < 0:
        raise ValueError("limit_molecules cannot be negative")

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

    graphs, failed = read_sdf_graphs(sdf_path, limit_molecules)
    print(f"Converted {len(graphs)} SDF records")
    splits = split_conformers(
        graphs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    split_sizes = ", ".join(f"{key}={len(value)}" for key, value in splits.items())
    print(f"Split sizes: {split_sizes}")

    save_graph_splits(splits, processed_path)
    print(f"Completed: output={processed_path}, failed_conversions={failed}")


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 3D SDF file to standard LoQI training artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sdf_path",
        required=True,
        help="Input SDF with one 3D conformer per record",
    )
    parser.add_argument(
        "--save_data_folder",
        required=True,
        help="Output dataset root; standard artifacts are written under processed/",
    )
    parser.add_argument(
        "--limit_molecules",
        type=int,
        default=0,
        help="Optional successful-record limit (0 means unlimited)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Process at most 30 valid SDF records",
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
